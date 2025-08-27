from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache, wraps
import tempfile
import logging
import pytz
import requests

# If you use g4f, keep import dynamic inside class to avoid import-time failures on deploy
# from g4f.client import Client

# ---------- Config ----------
VIETNAM_TZ = pytz.timezone(os.getenv("VIETNAM_TZ", "Asia/Ho_Chi_Minh"))

MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 80))
MAX_INPUT_MESSAGES = int(os.getenv("MAX_INPUT_MESSAGES", 3))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_CALL_TIMEOUT = float(os.getenv("API_CALL_TIMEOUT", 15.0))

# URL to POST history to (example: https://webhook.site/xxxx or Beeceptor URL or JSONBin endpoint)
STORAGE_API_URL = os.getenv("STORAGE_API_URL", "").rstrip("/")  # no trailing slash
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY", "")

# Control how often we attempt to push history externally
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", "3.0"))

# Thread pool for async background tasks
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chatbot")

# Logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ultrafast")
logger.setLevel(logging.WARNING)

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.logger.disabled = True

# Helper decorator to run function in background
def async_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return executor.submit(func, *args, **kwargs)
    return wrapper

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

        # lazy Client import to avoid import-time failures
        try:
            from g4f.client import Client
            self._client_available = True
            self.client = Client()
        except Exception:
            self._client_available = False
            self.client = None

        self.max_messages = max_messages
        self.messages = []  # in-memory store: list of {"role","content","timestamp"}
        self._message_cache = {}
        self._last_save_time = 0.0
        self._save_interval = SAVE_INTERVAL
        self._lock = threading.RLock()
        self._vietnam_tz = VIETNAM_TZ

        # Try to preload history from external URL (best-effort)
        self._load_history_external()

        self.initialized = True

    def _now_ts(self):
        return datetime.now(self._vietnam_tz).strftime("%Y-%m-%d %H:%M:%S")

    def get_vietnam_time_info(self):
        vn = datetime.now(self._vietnam_tz)
        weekdays = ['Thứ Hai','Thứ Ba','Thứ Tư','Thứ Năm','Thứ Sáu','Thứ Bảy','Chủ Nhật']
        weekday_vn = weekdays[vn.weekday()]
        months = ['Tháng 1','Tháng 2','Tháng 3','Tháng 4','Tháng 5','Tháng 6',
                  'Tháng 7','Tháng 8','Tháng 9','Tháng 10','Tháng 11','Tháng 12']
        month_vn = months[vn.month - 1]
        return {
            'current_time': vn.strftime('%H:%M:%S'),
            'current_date': f"{vn.day} {month_vn} năm {vn.year}",
            'weekday': weekday_vn,
            'full_datetime': f"{weekday_vn}, {vn.day} {month_vn} {vn.year} lúc {vn.strftime('%H:%M:%S')}",
            'timestamp': vn.timestamp(),
            'location': 'Khánh Hòa, Việt Nam'
        }

    def _trim_messages(self):
        if len(self.messages) > self.max_messages:
            with self._lock:
                system_msgs = [m for m in self.messages if m.get("role") == "system"]
                recent = [m for m in self.messages if m.get("role") != "system"][-int(self.max_messages*0.6):]
                self.messages = system_msgs + recent

    def clear_history(self):
        with self._lock:
            self.messages = []
            self._message_cache.clear()
        # optionally tell external store to clear (best-effort)
        if STORAGE_API_URL:
            try:
                headers = {}
                if STORAGE_API_KEY:
                    headers["Authorization"] = f"Bearer {STORAGE_API_KEY}"
                # Many simple webhook endpoints won't accept DELETE; best-effort
                executor.submit(lambda: requests.delete(STORAGE_API_URL, headers=headers, timeout=2))
            except Exception:
                pass

    def _build_minimal_payload(self, user_input):
        system_msgs = [{"role": m["role"], "content": m["content"]} for m in self.messages if m.get("role") == "system"]
        non_system = [{"role": m["role"], "content": m["content"]} for m in self.messages if m.get("role") in ("user","assistant")]
        recent = non_system[-MAX_INPUT_MESSAGES:] if non_system else []
        payload = system_msgs + recent + [{"role": "user", "content": user_input}]
        return payload

    def add_system_with_time(self):
        # remove old system messages and insert fresh one
        with self._lock:
            self.messages = [m for m in self.messages if m.get("role") != "system"]
            time_info = self.get_vietnam_time_info()
            system_content = f"Bạn là trợ lý AI tại {time_info['location']}. Thời gian hiện tại: {time_info['full_datetime']}."
            self.messages.insert(0, {"role":"system","content":system_content,"timestamp": self._now_ts()})

    # --- External storage interactions (best-effort) ---
    def _load_history_external(self):
        """Try to GET existing history from STORAGE_API_URL (if it supports GET). Best-effort only."""
        if not STORAGE_API_URL:
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
                        # load only last max_messages
                        with self._lock:
                            self.messages = data["messages"][-self.max_messages:]
                except Exception:
                    # if response is raw array
                    try:
                        data = r.json()
                        if isinstance(data, list):
                            with self._lock:
                                self.messages = data[-self.max_messages:]
                    except Exception:
                        pass
        except Exception:
            pass

    @async_task
    def save_history_async(self, force=False):
        """Push latest history to external URL (POST) non-blocking. Use short timeout."""
        current = time.time()
        if not force and (current - self._last_save_time) < self._save_interval:
            return
        with self._lock:
            try:
                # trim aggressively before sending
                if len(self.messages) > self.max_messages:
                    self._trim_messages()

                payload = {"messages": self.messages[-self.max_messages:]}
                headers = {"Content-Type": "application/json"}
                if STORAGE_API_KEY:
                    headers["Authorization"] = f"Bearer {STORAGE_API_KEY}"

                if STORAGE_API_URL:
                    # POST to STORAGE_API_URL (best-effort, short timeout)
                    try:
                        requests.post(STORAGE_API_URL, json=payload, headers=headers, timeout=2)
                    except Exception:
                        # try fallback: post to STORAGE_API_URL + /history if some endpoints expect path
                        try:
                            requests.post(f"{STORAGE_API_URL.rstrip('/')}/history", json=payload, headers=headers, timeout=2)
                        except Exception:
                            pass
                self._last_save_time = current
            except Exception:
                pass

    # --- Core LLM call (ultra-minimal payload) ---
    def get_response_ultra_fast(self, user_input):
        start = time.time()
        try:
            # ensure system msg exists
            if not any(m.get("role") == "system" for m in self.messages) or len(self.messages) % 5 == 0:
                self.add_system_with_time()

            payload_messages = self._build_minimal_payload(user_input)

            bot_reply = None
            # call g4f if available, else fallback
            if self._client_available and self.client:
                try:
                    response = self.client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=payload_messages,
                        max_tokens=600,
                        temperature=0.5,
                        top_p=0.9,
                        timeout=API_CALL_TIMEOUT
                    )
                    try:
                        bot_reply = response.choices[0].message.content.strip()
                    except Exception:
                        bot_reply = str(response).strip()
                except Exception:
                    bot_reply = None

            if not bot_reply or len(bot_reply) < 2:
                bot_reply = "Xin lỗi, hiện tại tôi không truy cập được mô-đun trả lời nhanh. Bạn có thể thử lại hoặc chờ một chút."

            # append into history
            ts = self._now_ts()
            with self._lock:
                self.messages.append({"role":"user","content":user_input,"timestamp":ts})
                self.messages.append({"role":"assistant","content":bot_reply,"timestamp":ts})
                # manage memory
                if len(self.messages) > int(self.max_messages * 0.8):
                    self._trim_messages()

            # save asynchronously (best-effort)
            try:
                self.save_history_async()
            except Exception:
                pass

            elapsed = time.time() - start
            if elapsed > 10:
                logger.warning(f"Slow response: {elapsed:.2f}s")

            return bot_reply
        except Exception:
            # on error, still try to record user message
            try:
                with self._lock:
                    self.messages.append({"role":"user","content":user_input,"timestamp": self._now_ts()})
                self.save_history_async()
            except:
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

        # time shortcut
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

        # otherwise ask LLM (or fallback)
        reply = bot.get_response_ultra_fast(message)
        return jsonify({"ok": True, "reply": reply})
    except Exception:
        return jsonify({"ok": False, "error": "Server busy"}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        return jsonify({"ok": True, "messages": bot.messages[-30:]})
    except:
        return jsonify({"ok": True, "messages": []})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        bot.clear_history()
        return jsonify({"ok": True, "message": "Cleared"})
    except:
        return jsonify({"ok": False, "error": "Error"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        t = bot.get_vietnam_time_info()
        return jsonify({"ok": True, "status":"online", "messages_count": len(bot.messages), "vietnam_time": t['current_time']})
    except:
        return jsonify({"ok": True, "status":"online"})

@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.errorhandler(500)
def internal_err(e):
    return jsonify({"ok": False, "error": "Server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
