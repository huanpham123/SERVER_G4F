# index.py
import os
import json
import time
import tempfile
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Optional redis (Upstash). If not available, redis_client stays None.
try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    redis = None
    REDIS_AVAILABLE = False

# Optional g4f client (your LLM client). If not available, llm client will raise.
try:
    from g4f.client import Client as G4FClient
    G4F_AVAILABLE = True
except Exception:
    G4FClient = None
    G4F_AVAILABLE = False

# -----------------------
# Config (env)
# -----------------------
HISTORY_KEY = os.environ.get("HISTORY_KEY", "chat_history_v1")
HISTORY_FILE = os.environ.get("HISTORY_FILE", "/tmp/chat_history.json")  # ephemeral on serverless
REDIS_URL = os.environ.get("REDIS_URL", "")  # Upstash-style URL if provided
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_CALL_TIMEOUT = float(os.environ.get("API_CALL_TIMEOUT", "12.0"))  # seconds
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "120"))
MAX_INPUT_MESSAGES = int(os.environ.get("MAX_INPUT_MESSAGES", "3"))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "180"))  # seconds for identical query cache

# -----------------------
# Setup logging
# -----------------------
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fastchat")
logger.setLevel(logging.INFO)

# -----------------------
# Redis client (optional)
# -----------------------
redis_client = None
if REDIS_URL and REDIS_AVAILABLE:
    try:
        # support both redis:// and upstash-style
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        # quick ping
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning("Redis not available: %s", e)
        redis_client = None

# -----------------------
# LLM wrapper + thread executor
# -----------------------
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")

class LLMWrapper:
    def __init__(self):
        if not G4F_AVAILABLE:
            logger.warning("g4f client not available; LLM calls will fail unless installed.")
        self.client = G4FClient() if G4F_AVAILABLE else None

    def call(self, messages, timeout=API_CALL_TIMEOUT):
        """
        Blocking call to LLM. Designed to run in thread executor.
        Returns reply string or raises.
        """
        if not self.client:
            raise RuntimeError("LLM client not installed/available.")

        def _call():
            # Defensive wrapper around g4f usage
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=800,
                temperature=0.5,
                top_p=0.9
            )
            # Try to extract the common structure
            try:
                return resp.choices[0].message.content.strip()
            except Exception:
                # fallback to string
                return str(resp).strip()

        fut = executor.submit(_call)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            fut.cancel()
            raise TimeoutError("LLM call timed out")
        except Exception as e:
            raise

llm = LLMWrapper()

# -----------------------
# Utility: history and cache management
# -----------------------
def _local_save_history(history):
    try:
        dirpath = os.path.dirname(HISTORY_FILE) or "/tmp"
        with tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False, encoding="utf-8") as tf:
            json.dump(history[-MAX_HISTORY:], tf, ensure_ascii=False, separators=(",", ":"))
            tmpname = tf.name
        os.replace(tmpname, HISTORY_FILE)
        return True
    except Exception as e:
        logger.warning("Local save history failed: %s", e)
        return False

def _local_load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    return data[-MAX_HISTORY:]
    except Exception as e:
        logger.warning("Local load history failed: %s", e)
    return []

def _redis_set_history(history):
    if not redis_client:
        return False
    try:
        redis_client.set(HISTORY_KEY, json.dumps(history[-MAX_HISTORY:], ensure_ascii=False), ex=None)
        return True
    except Exception as e:
        logger.warning("Redis set history failed: %s", e)
        return False

def _redis_get_history():
    if not redis_client:
        return None
    try:
        v = redis_client.get(HISTORY_KEY)
        if v:
            return json.loads(v)
    except Exception as e:
        logger.warning("Redis get history failed: %s", e)
    return None

def save_history_async(history):
    """Non-blocking save (background via executor)"""
    def _save():
        if redis_client:
            ok = _redis_set_history(history)
            if ok:
                return
        _local_save_history(history)
    try:
        executor.submit(_save)
    except Exception:
        _save()

def load_history():
    if redis_client:
        v = _redis_get_history()
        if v is not None:
            return v
    return _local_load_history()

# simple cache for identical messages (in-Redis if available)
def get_cached_reply(key_hash):
    if redis_client:
        try:
            return redis_client.get(key_hash)
        except Exception:
            return None
    return None

def set_cached_reply(key_hash, reply):
    if redis_client:
        try:
            redis_client.set(key_hash, reply, ex=CACHE_TTL)
        except Exception:
            pass

# -----------------------
# Helpers: payload & fallback
# -----------------------
def system_prompt_with_time():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"role": "system", "content": f"Bạn là một trợ lý AI. Time (UTC+7): {now}"}

def build_payload(history, user_input):
    system_msgs = [m for m in history if m.get("role") == "system"]
    conv = [m for m in history if m.get("role") in ("user", "assistant")]
    recent = conv[-MAX_INPUT_MESSAGES:] if conv else []
    payload = []
    if system_msgs:
        payload.extend([{"role":m["role"], "content":m["content"]} for m in system_msgs])
    else:
        payload.append(system_prompt_with_time())
    for m in recent:
        payload.append({"role": m["role"], "content": m["content"]})
    payload.append({"role": "user", "content": user_input})
    return payload

def fallback_reply(user_input):
    # Avoid the specific phrase "Hệ thống đang bận"
    u = user_input.strip()
    if not u:
        return "Bạn chưa nhập gì. Vui lòng nhập câu hỏi."
    if len(u) < 60:
        return "Mình nhận được câu hỏi của bạn — hiện tạm trả lời ngắn: hãy hỏi thêm chi tiết để được trả lời đầy đủ."
    return "Xin lỗi, hiện chưa thể trả lời đầy đủ. Vui lòng thử hỏi ngắn hơn hoặc thử lại."

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# quick health root -> sv.html
@app.route("/")
def index():
    try:
        return render_template("sv.html")
    except Exception:
        return "<h2>Template sv.html not found in templates/</h2>", 404

# Chat endpoint
@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"ok": False, "error": "Invalid JSON"}), 400
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Empty message"}), 400
        if len(message) > 5000:
            return jsonify({"ok": False, "error": "Message too long"}), 400

        # quick commands
        if message.lower() in ("clear", "/clear", "xóa", "xoa"):
            # clear both local and redis
            if os.path.exists(HISTORY_FILE):
                try:
                    os.remove(HISTORY_FILE)
                except:
                    pass
            if redis_client:
                try:
                    redis_client.delete(HISTORY_KEY)
                except:
                    pass
            return jsonify({"ok": True, "reply": "Đã xóa lịch sử chat."})

        # load history
        history = load_history() or []

        # cache check (hash key)
        cache_key = f"cache:reply:{hash(message)}"
        cached = get_cached_reply(cache_key)
        if cached:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history.append({"role":"user","content":message,"timestamp":ts})
            history.append({"role":"assistant","content":cached,"timestamp":ts})
            save_history_async(history)
            return jsonify({"ok": True, "reply": cached})

        # Build payload
        payload = build_payload(history, message)

        # call LLM in thread with timeout
        try:
            # we run blocking call in separate thread to allow timeout
            future = executor.submit(lambda: llm.call(payload, timeout=API_CALL_TIMEOUT))
            reply = future.result(timeout=API_CALL_TIMEOUT + 1.0)  # small margin
            if not reply or len(reply.strip()) < 2:
                reply = fallback_reply(message)
        except FuturesTimeout:
            # cancel and fallback
            try:
                future.cancel()
            except:
                pass
            reply = fallback_reply(message)
        except TimeoutError:
            reply = fallback_reply(message)
        except Exception as e:
            logger.warning("LLM call exception: %s", e)
            reply = fallback_reply(message)

        # persist reply to history (non-blocking)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history.append({"role":"user","content":message,"timestamp":ts})
        history.append({"role":"assistant","content":reply,"timestamp":ts})
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        save_history_async(history)

        # cache short-lived if redis present
        if redis_client:
            try:
                redis_client.set(cache_key, reply, ex=CACHE_TTL)
            except Exception:
                pass

        return jsonify({"ok": True, "reply": reply})
    except Exception as e:
        logger.exception("api_chat error")
        # fallback safe reply (do not return the "Hệ thống đang bận" phrase)
        reply = fallback_reply("")
        return jsonify({"ok": True, "reply": reply, "info": "fallback_due_to_error"})

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        history = load_history() or []
        return jsonify({"ok": True, "messages": history[-50:]})
    except Exception:
        return jsonify({"ok": True, "messages": []})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        if os.path.exists(HISTORY_FILE):
            try:
                os.remove(HISTORY_FILE)
            except:
                pass
        if redis_client:
            try:
                redis_client.delete(HISTORY_KEY)
            except:
                pass
        return jsonify({"ok": True, "message": "Cleared"})
    except Exception:
        return jsonify({"ok": False, "error": "Error clearing"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        redis_ok = False
        if redis_client:
            try:
                redis_client.ping()
                redis_ok = True
            except:
                redis_ok = False
        return jsonify({
            "ok": True,
            "status": "online",
            "model": MODEL_NAME,
            "redis": redis_ok,
            "api_call_timeout": API_CALL_TIMEOUT
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Local dev
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # threaded True to allow concurrent requests in dev; Vercel serverless will handle concurrency separately
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
