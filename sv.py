# api/index.py
import os
import time
import logging
import threading
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# optional libs (we'll import lazily inside functions)
try:
    import pytz
except Exception:
    pytz = None

try:
    import requests
except Exception:
    requests = None

# ---------- Config ----------
VIETNAM_TZ = pytz.timezone(os.getenv("VIETNAM_TZ", "Asia/Ho_Chi_Minh")) if pytz else None

MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 80))
MAX_INPUT_MESSAGES = int(os.getenv("MAX_INPUT_MESSAGES", 3))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# Default timeout: keep conservative so serverless will respond quickly.
API_CALL_TIMEOUT = float(os.getenv("API_CALL_TIMEOUT", 10.0))

STORAGE_API_URL = os.getenv("STORAGE_API_URL", "").rstrip("/")
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY", "")

SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", "3.0"))

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="chatbot")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ultrafast")
logger.setLevel(logging.INFO)

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Helper decorator to run function in background
def async_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return executor.submit(func, *args, **kwargs)
    return wrapper

# ---------- Utility ----------
def _now_ts(tz=VIETNAM_TZ):
    if tz:
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _get_vietnam_time_info():
    if VIETNAM_TZ:
        vn = datetime.now(VIETNAM_TZ)
    else:
        vn = datetime.utcnow()
    weekdays = ['Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật']
    weekday_vn = weekdays[vn.weekday()] if 0 <= vn.weekday() < 7 else ""
    months = ['Tháng 1','Tháng 2','Tháng 3','Tháng 4','Tháng 5','Tháng 6',
              'Tháng 7','Tháng 8','Tháng 9','Tháng 10','Tháng 11','Tháng 12']
    month_vn = months[vn.month - 1] if 1 <= vn.month <= 12 else f"Tháng {vn.month}"
    return {
        'current_time': vn.strftime('%H:%M:%S'),
        'current_date': f"{vn.day} {month_vn} năm {vn.year}",
        'weekday': weekday_vn,
        'full_datetime': f"{weekday_vn}, {vn.day} {month_vn} {vn.year} lúc {vn.strftime('%H:%M:%S')}",
        'timestamp': vn.timestamp(),
        'location': 'Khánh Hòa, Việt Nam'
    }

# ---------- Bot Implementation ----------
class UltraFastChatBot:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_messages=MAX_MESSAGES):
        if getattr(self, "initialized", False):
            return

        self.max_messages = max_messages
        self.messages = []  # list of {"role":..., "content":..., "timestamp":...}
        self._message_cache = {}
        self._last_save_time = 0.0
        self._save_interval = SAVE_INTERVAL
        self._lock = threading.RLock()

        # Do NOT import heavy libs at module import time. Will be lazy imported when needed.
        self._g4f_client = None
        self.initialized = True

        # try to preload history (best-effort)
        try:
            self._load_history_external()
        except Exception as e:
            logger.debug("No external history loaded: %s", e)

    def _now_ts(self):
        return _now_ts(VIETNAM_TZ)

    def get_vietnam_time_info(self):
        return _get_vietnam_time_info()

    def _trim_messages(self):
        with self._lock:
            if len(self.messages) > self.max_messages:
                system_msgs = [m for m in self.messages if m.get("role") == "system"]
                recent = [m for m in self.messages if m.get("role") != "system"][-int(self.max_messages*0.6):]
                self.messages = system_msgs + recent

    def clear_history(self):
        with self._lock:
            self.messages = []
            self._message_cache.clear()
        # best-effort delete on external storage
        if STORAGE_API_URL and requests:
            try:
                headers = {}
                if STORAGE_API_KEY:
                    headers["Authorization"] = f"Bearer {STORAGE_API_KEY}"
                executor.submit(lambda: requests.delete(STORAGE_API_URL, headers=headers, timeout=2))
            except Exception:
                pass

    def _build_minimal_payload(self, user_input):
        system_msgs = [{"role": m["role"], "content": m["content"]} for m in self.messages if m.get("role") == "system"]
        non_system = [{"role": m["role"], "content": m["content"]} for m in self.messages if m.get("role") in ("user","assistant")]
        recent = non_system[-MAX_INPUT_MESSAGES:] if non_system else []
        return system_msgs + recent + [{"role": "user", "content": user_input}]

    def add_system_with_time(self):
        with self._lock:
            self.messages = [m for m in self.messages if m.get("role") != "system"]
            time_info = self.get_vietnam_time_info()
            system_content = f"Bạn là trợ lý AI tại {time_info['location']}. Thời gian hiện tại: {time_info['full_datetime']}."
            self.messages.insert(0, {"role":"system","content":system_content,"timestamp": self._now_ts()})

    def _load_history_external(self):
        if not STORAGE_API_URL or not requests:
            return
        try:
            headers = {}
            if STORAGE_API_KEY:
                headers["Authorization"] = f"Bearer {STORAGE_API_KEY}"
            r = requests.get(STORAGE_API_URL, headers=headers, timeout=2)
            if r.status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
                        with self._lock:
                            self.messages = data["messages"][-self.max_messages:]
                    elif isinstance(data, list):
                        with self._lock:
                            self.messages = data[-self.max_messages:]
                except Exception:
                    pass
        except Exception:
            pass

    @async_task
    def save_history_async(self, force=False):
        current = time.time()
        if not force and (current - self._last_save_time) < self._save_interval:
            return
        with self._lock:
            try:
                if len(self.messages) > self.max_messages:
                    self._trim_messages()
                payload = {"messages": self.messages[-self.max_messages:]}
                headers = {"Content-Type": "application/json"}
                if STORAGE_API_KEY:
                    headers["Authorization"] = f"Bearer {STORAGE_API_KEY}"
                if STORAGE_API_URL and requests:
                    try:
                        requests.post(STORAGE_API_URL, json=payload, headers=headers, timeout=2)
                    except Exception:
                        try:
                            requests.post(f"{STORAGE_API_URL.rstrip('/')}/history", json=payload, headers=headers, timeout=2)
                        except Exception:
                            pass
                self._last_save_time = current
            except Exception:
                pass

    # --- LLM caller (lazy & safe) ---
    def _llm_call_once(self, messages_payload):
        """Try to get a response from available backends (g4f -> OpenAI via HTTP -> None)."""
        # 1) Try g4f if available
        try:
            # lazy import & client init
            if self._g4f_client is None:
                try:
                    from g4f.client import Client as G4FClient
                    self._g4f_client = G4FClient()
                except Exception:
                    self._g4f_client = False  # mark as unavailable
            if self._g4f_client and self._g4f_client is not False:
                try:
                    # use chat completions interface if available
                    resp = self._g4f_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages_payload,
                        max_tokens=600,
                        temperature=0.5,
                        top_p=0.9,
                        timeout=API_CALL_TIMEOUT
                    )
                    try:
                        return resp.choices[0].message.content.strip()
                    except Exception:
                        return str(resp).strip()
                except Exception as e:
                    logger.info("g4f call failed: %s", e)
        except Exception:
            pass

        # 2) Try OpenAI via HTTP if OPENAI_API_KEY provided (lazy)
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        if OPENAI_API_KEY and requests:
            try:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                body = {
                    "model": MODEL_NAME or "gpt-3.5-turbo",
                    "messages": messages_payload,
                    "max_tokens": 600,
                    "temperature": 0.5,
                }
                r = requests.post(url, json=body, headers=headers, timeout=max(3.0, API_CALL_TIMEOUT - 1.0))
                r.raise_for_status()
                j = r.json()
                choices = j.get("choices") or []
                if choices:
                    txt = choices[0].get("message", {}).get("content") or choices[0].get("text")
                    return (txt or "").strip()
            except Exception as e:
                logger.info("OpenAI HTTP call failed: %s", e)

        # 3) No backend available or all failed
        return None

    def _generate_quick_reply(self, user_input):
        """Produce an immediate helpful reply when external LLM is unavailable or slow."""
        # Heuristics: if user asks time-like questions, answer precisely.
        l = user_input.lower()
        time_keywords = ["giờ","thời gian","bây giờ","hiện tại","mấy giờ"]
        if any(k in l for k in time_keywords):
            tinfo = self.get_vietnam_time_info()
            return f"Hiện tại là {tinfo['full_datetime']} tại {tinfo['location']}."
        # If user asks to clear
        if l.strip() in ("clear","/clear","xóa","xoa"):
            self.clear_history()
            return "Đã xóa lịch sử chat."
        # Try to use last assistant message as quick context
        with self._lock:
            last_assistant = next((m for m in reversed(self.messages) if m.get("role") == "assistant"), None)
        if last_assistant:
            snippet = last_assistant.get("content","")
            return f"Mình nhận được câu hỏi của bạn: \"{user_input}\". Dựa trên phản hồi trước đây, tóm tắt nhanh: {snippet[:500]}"
        # Default friendly fallback
        return ("Mình đã nhận câu hỏi của bạn: \"{}\". Hiện tại dịch vụ trả lời đang chậm hoặc không liên lạc được với mô-đun LLM ngoài. "
                "Tạm thời mình trả lời ngắn gọn và sẽ lưu câu hỏi để xử lý kỹ hơn sau.").format(user_input[:800])

    def get_response_ultra_fast(self, user_input):
        start = time.time()
        try:
            # ensure system msg exists occasionally
            if not any(m.get("role") == "system" for m in self.messages) or len(self.messages) % 5 == 0:
                self.add_system_with_time()

            payload_messages = self._build_minimal_payload(user_input)

            # Submit llm call to thread pool and wait with timeout
            future = executor.submit(self._llm_call_once, payload_messages)
            llm_reply = None
            try:
                llm_reply = future.result(timeout=API_CALL_TIMEOUT)
            except TimeoutError:
                logger.warning("LLM call timed out after %.2fs", API_CALL_TIMEOUT)
                # attempt to cancel (best-effort)
                try:
                    future.cancel()
                except Exception:
                    pass
                llm_reply = None
            except Exception as exc:
                logger.exception("LLM call raised exception: %s", exc)
                llm_reply = None

            # If got a good reply from LLM, use it
            if llm_reply and len(llm_reply.strip()) >= 2:
                bot_reply = llm_reply.strip()
            else:
                # produce an immediate helpful reply (not the generic error)
                bot_reply = self._generate_quick_reply(user_input)
                # and start an async background attempt to get a fuller reply and save it into history
                @async_task
                def background_fill(orig_payload, original_input):
                    try:
                        fuller = self._llm_call_once(orig_payload)
                        if fuller and len(str(fuller).strip()) > 1:
                            ts = self._now_ts()
                            with self._lock:
                                # append user + assistant (fuller) replacing the last quick fallback assistant
                                self.messages.append({"role":"user","content": original_input, "timestamp": ts})
                                self.messages.append({"role":"assistant","content": str(fuller).strip(), "timestamp": ts})
                                if len(self.messages) > int(self.max_messages * 0.8):
                                    self._trim_messages()
                            # also try to save history externally
                            try:
                                self.save_history_async(force=True)
                            except Exception:
                                pass
                    except Exception as e:
                        logger.info("Background fuller LLM attempt failed: %s", e)
                try:
                    background_fill(payload_messages, user_input)
                except Exception:
                    pass

            # append immediate reply to history
            ts = self._now_ts()
            with self._lock:
                self.messages.append({"role":"user","content":user_input,"timestamp":ts})
                self.messages.append({"role":"assistant","content":bot_reply,"timestamp":ts})
                if len(self.messages) > int(self.max_messages * 0.8):
                    self._trim_messages()

            # save asynchronously (best-effort)
            try:
                self.save_history_async()
            except Exception:
                pass

            elapsed = time.time() - start
            if elapsed > 8:
                logger.warning("Slow overall response: %.2fs", elapsed)

            return bot_reply

        except Exception as e:
            logger.exception("get_response_ultra_fast unexpected error: %s", e)
            try:
                with self._lock:
                    self.messages.append({"role":"user","content":user_input,"timestamp": self._now_ts()})
                self.save_history_async()
            except Exception:
                pass
            return "Hệ thống đang bận, vui lòng thử lại sau ít phút."

# Global bot
bot = UltraFastChatBot()

# ---------- Routes ----------
@app.route("/")
def index():
    try:
        return render_template("sv.html")
    except Exception:
        return "<h3>UI not found — please add templates/sv.html</h3>", 404

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        if not request.is_json:
            return jsonify({"ok": False, "error": "Invalid content type"}), 400
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"ok": False, "error": "Invalid JSON"}), 400

        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Empty message"}), 400
        if len(message) > 1500:
            return jsonify({"ok": False, "error": "Message too long"}), 400

        # quick clear commands
        if message.lower() in ("clear","/clear","xóa","xoa"):
            bot.clear_history()
            return jsonify({"ok": True, "reply": "Đã xóa lịch sử chat."})

        # time shortcut handled by bot
        time_keywords = ["giờ","thời gian","ngày","tháng","năm","bây giờ","hiện tại"]
        if any(k in message.lower() for k in time_keywords):
            time_info = bot.get_vietnam_time_info()
            reply = f"Hiện tại là {time_info['full_datetime']} tại {time_info['location']}."
            ts = bot._now_ts()
            with bot._lock:
                bot.messages.append({"role":"user","content":message,"timestamp":ts})
                bot.messages.append({"role":"assistant","content":reply,"timestamp":ts})
            bot.save_history_async()
            return jsonify({"ok": True, "reply": reply})

        # otherwise ask LLM (or fallback quick answer)
        reply = bot.get_response_ultra_fast(message)
        return jsonify({"ok": True, "reply": reply})
    except Exception as e:
        logger.exception("api_chat error: %s", e)
        return jsonify({"ok": False, "error": "Server busy"}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        return jsonify({"ok": True, "messages": bot.messages[-30:]})
    except Exception:
        return jsonify({"ok": True, "messages": []})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        bot.clear_history()
        return jsonify({"ok": True, "message": "Cleared"})
    except Exception:
        return jsonify({"ok": False, "error": "Error"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        t = bot.get_vietnam_time_info()
        return jsonify({"ok": True, "status":"online", "messages_count": len(bot.messages), "vietnam_time": t['current_time']})
    except Exception:
        return jsonify({"ok": True, "status":"online"})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.errorhandler(500)
def internal_err(e):
    return jsonify({"ok": False, "error": "Server error"}), 500

# For local dev only
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
