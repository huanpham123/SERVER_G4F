# api/index.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
import logging

# lazy imports at runtime (to avoid import-time failures on deploy)
try:
    import pytz
except Exception:
    pytz = None

try:
    import requests
except Exception:
    requests = None

# Config
VIETNAM_TZ = os.getenv("VIETNAM_TZ", "Asia/Ho_Chi_Minh")
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "80"))
MAX_INPUT_MESSAGES = int(os.getenv("MAX_INPUT_MESSAGES", "3"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_CALL_TIMEOUT = float(os.getenv("API_CALL_TIMEOUT", "15.0"))
STORAGE_API_URL = os.getenv("STORAGE_API_URL", "").rstrip("/")
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY", "")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", "3.0"))

# Logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ultrafast")
logger.setLevel(logging.WARNING)

# Flask app (Vercel picks up `app`)
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Thread pool for background tasks (kept but used best-effort)
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="chatbot")

def async_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return executor.submit(func, *args, **kwargs)
    return wrapper

class UltraFastChatBot:
    """Singleton chat-bot. Lazy imports for optional libs (g4f)."""
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

        # lazy import of g4f (if available) — won't fail deploy if missing
        try:
            from g4f.client import Client
            self._client_available = True
            self.client = Client()
        except Exception:
            self._client_available = False
            self.client = None

        self.max_messages = max_messages
        self.messages = []
        self._last_save_time = 0.0
        self._save_interval = SAVE_INTERVAL
        self._lock = threading.RLock()

        # timezone helper
        if pytz:
            try:
                self._vietnam_tz = pytz.timezone(VIETNAM_TZ)
            except Exception:
                self._vietnam_tz = None
        else:
            self._vietnam_tz = None

        # do NOT call external network at import time; keep load external as best-effort method executed later
        self.initialized = True

    def _now_ts(self):
        if self._vietnam_tz:
            return datetime.now(self._vietnam_tz).strftime("%Y-%m-%d %H:%M:%S")
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    def get_vietnam_time_info(self):
        if self._vietnam_tz:
            vn = datetime.now(self._vietnam_tz)
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

    def _trim_messages(self):
        with self._lock:
            if len(self.messages) > self.max_messages:
                system_msgs = [m for m in self.messages if m.get("role") == "system"]
                non_system = [m for m in self.messages if m.get("role") != "system"]
                recent = non_system[-int(self.max_messages * 0.6):]
                self.messages = system_msgs + recent

    def clear_history(self):
        with self._lock:
            self.messages = []
        # best-effort external delete
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

    def get_response_ultra_fast(self, user_input):
        try:
            if not any(m.get("role") == "system" for m in self.messages) or len(self.messages) % 5 == 0:
                self.add_system_with_time()

            payload_messages = self._build_minimal_payload(user_input)

            bot_reply = None
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

            ts = self._now_ts()
            with self._lock:
                self.messages.append({"role":"user","content":user_input,"timestamp":ts})
                self.messages.append({"role":"assistant","content":bot_reply,"timestamp":ts})
                if len(self.messages) > int(self.max_messages * 0.8):
                    self._trim_messages()

            try:
                self.save_history_async()
            except Exception:
                pass

            return bot_reply
        except Exception:
            try:
                with self._lock:
                    self.messages.append({"role":"user","content":user_input,"timestamp": self._now_ts()})
                self.save_history_async()
            except:
                pass
            return "Hệ thống đang bận, vui lòng thử lại sau ít phút."

# lazy global bot
_bot_instance = None
_bot_lock = threading.Lock()

def get_bot():
    global _bot_instance
    if _bot_instance is None:
        with _bot_lock:
            if _bot_instance is None:
                _bot_instance = UltraFastChatBot()
    return _bot_instance

# Routes
@app.route("/")
def index():
    return render_template("sv.html")

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

        bot = get_bot()

        if message.lower() in ("clear","/clear","xóa","xoa"):
            bot.clear_history()
            return jsonify({"ok": True, "reply": "Đã xóa lịch sử chat."})

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

        reply = bot.get_response_ultra_fast(message)
        return jsonify({"ok": True, "reply": reply})
    except Exception as e:
        logger.exception("api_chat error")
        return jsonify({"ok": False, "error": "Server busy"}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        bot = get_bot()
        return jsonify({"ok": True, "messages": bot.messages[-30:]})
    except Exception:
        return jsonify({"ok": True, "messages": []})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        bot = get_bot()
        bot.clear_history()
        return jsonify({"ok": True, "message": "Cleared"})
    except Exception:
        return jsonify({"ok": False, "error": "Error"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        bot = get_bot()
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

# For local dev only
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
