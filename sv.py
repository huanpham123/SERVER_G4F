#!/usr/bin/env python3
# server.py
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
import logging
import pytz

# ---------------- Config ----------------
VIETNAM_TZ = pytz.timezone(os.getenv("VIETNAM_TZ", "Asia/Ho_Chi_Minh"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", 80))
MAX_INPUT_MESSAGES = int(os.getenv("MAX_INPUT_MESSAGES", 3))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_CALL_TIMEOUT = float(os.getenv("API_CALL_TIMEOUT", 15.0))

# Thread pool for light background tasks
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chatbot")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ultrafast")
logger.setLevel(logging.INFO)

# Flask app
app = Flask(__name__, template_folder="templates")
# Allow all origins for /api/*
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.logger.disabled = True

# Helper: run function in background
def async_task(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return executor.submit(func, *args, **kwargs)
    return wrapper

# ---------------- Bot Class ----------------
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

        # lazy import g4f
        try:
            import g4f  # noqa: F401
            self._client_available = True
        except Exception as e:
            logger.warning(f"G4F import failed (will use fallback): {e}")
            self._client_available = False

        self.max_messages = max_messages
        self.messages = []  # list of {"role","content","timestamp"}
        self._message_cache = {}
        self._last_save_time = 0.0
        self._save_interval = 3.0
        self._lock = threading.RLock()
        self._vietnam_tz = VIETNAM_TZ
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

    def _build_minimal_payload(self, user_input):
        system_msgs = [{"role": m["role"], "content": m["content"]} for m in self.messages if m.get("role") == "system"]
        non_system = [{"role": m["role"], "content": m["content"]} for m in self.messages if m.get("role") in ("user","assistant")]
        recent = non_system[-MAX_INPUT_MESSAGES:] if non_system else []
        payload = system_msgs + recent + [{"role": "user", "content": user_input}]
        return payload

    def add_system_with_time(self):
        with self._lock:
            self.messages = [m for m in self.messages if m.get("role") != "system"]
            time_info = self.get_vietnam_time_info()
            system_content = f"Bạn là trợ lý AI tại {time_info['location']}. Thời gian hiện tại: {time_info['full_datetime']}."
            self.messages.insert(0, {"role":"system","content":system_content,"timestamp": self._now_ts()})

    def get_response_ultra_fast(self, user_input):
        start = time.time()
        try:
            # ensure system message exists periodically
            if not any(m.get("role") == "system" for m in self.messages) or len(self.messages) % 5 == 0:
                self.add_system_with_time()

            payload_messages = self._build_minimal_payload(user_input)

            bot_reply = None
            if self._client_available:
                try:
                    import g4f
                    response = g4f.ChatCompletion.create(
                        model=MODEL_NAME,
                        messages=payload_messages,
                        max_tokens=600,
                        temperature=0.5,
                        top_p=0.9,
                        timeout=API_CALL_TIMEOUT
                    )
                    if isinstance(response, str):
                        bot_reply = response.strip()
                    else:
                        bot_reply = str(response).strip()
                except Exception as e:
                    logger.error(f"G4F error: {e}", exc_info=True)
                    bot_reply = None

            if not bot_reply or len(bot_reply) < 2:
                bot_reply = "Xin lỗi, hiện tại tôi không truy cập được mô-đun trả lời nhanh. Bạn có thể thử lại hoặc chờ một chút."

            ts = self._now_ts()
            with self._lock:
                self.messages.append({"role":"user","content":user_input,"timestamp":ts})
                self.messages.append({"role":"assistant","content":bot_reply,"timestamp":ts})
                if len(self.messages) > int(self.max_messages * 0.8):
                    self._trim_messages()

            elapsed = time.time() - start
            if elapsed > 10:
                logger.warning(f"Slow response: {elapsed:.2f}s")
            return bot_reply
        except Exception as e:
            logger.error(f"Error in get_response_ultra_fast: {e}", exc_info=True)
            try:
                with self._lock:
                    self.messages.append({"role":"user","content":user_input,"timestamp": self._now_ts()})
            except:
                pass
            return "Hệ thống đang bận, vui lòng thử lại sau ít phút."

# Global bot
bot = UltraFastChatBot()

# ---------------- Helpers / Routes ----------------
def json_response(payload, status=200):
    resp = make_response(json.dumps(payload, ensure_ascii=False), status)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    resp.headers["Connection"] = "close"
    return resp

@app.before_request
def log_request_brief():
    # Lightweight logging: show method and path. Avoid logging bodies for all requests to reduce noise,
    # but for /api/chat we will log body inside handler.
    logger.debug(f"Incoming request: {request.method} {request.path}")

@app.route("/")
def index():
    try:
        return render_template("sv.html")
    except Exception as e:
        logger.error(f"Render template error: {e}", exc_info=True)
        return f"<h3>UI not found — {e}</h3>", 404

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_chat():
    if request.method == "OPTIONS":
        # Reply simple preflight
        return json_response({"ok": True, "msg": "preflight"}, 200)

    try:
        # Log headers (useful to debug clients like ESP32)
        try:
            hdrs = dict(request.headers)
            logger.info(f"/api/chat headers: {hdrs}")
        except Exception:
            logger.warning("Could not read request headers fully")

        # Try to read JSON safely
        data = None
        try:
            data = request.get_json(force=False, silent=True)
        except Exception as e:
            logger.warning(f"get_json() exception: {e}")

        raw_body = ""
        try:
            raw_body = request.get_data(as_text=True) or ""
        except Exception as e:
            logger.warning(f"get_data() exception: {e}")

        logger.info(f"/api/chat raw_body_len={len(raw_body)} preview={raw_body[:800]}")

        # Support form data
        if not data and request.form:
            data = request.form.to_dict()

        # If still empty but raw present: try parse JSON or fallback to treat raw as message
        if not data and raw_body:
            try:
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                data = {"message": raw_body.strip()}

        if not data:
            return json_response({"ok": False, "error": "Invalid JSON or empty body"}, 400)

        message = (data.get("message") or "").strip() if isinstance(data, dict) else ""
        if not message:
            return json_response({"ok": False, "error": "Empty message"}, 400)
        if len(message) > 1500:
            return json_response({"ok": False, "error": "Message too long"}, 400)

        # Quick commands
        if message.lower() in ("clear","/clear","xóa","xoa"):
            bot.clear_history()
            return json_response({"ok": True, "reply": "Đã xóa lịch sử chat."}, 200)

        # Time shortcut handled locally
        time_keywords = ["giờ","thời gian","ngày","tháng","năm","bây giờ","hiện tại"]
        if any(k in message.lower() for k in time_keywords):
            time_info = bot.get_vietnam_time_info()
            reply = f"Hiện tại là {time_info['full_datetime']} tại {time_info['location']}."
            ts = bot._now_ts()
            with bot._lock:
                bot.messages.append({"role":"user","content":message,"timestamp":ts})
                bot.messages.append({"role":"assistant","content":reply,"timestamp":ts})
            return json_response({"ok": True, "reply": reply}, 200)

        # Normal LLM call (wrapped)
        try:
            reply = bot.get_response_ultra_fast(message)
            if not reply:
                reply = "Xin lỗi, hệ thống tạm thời bận. Vui lòng thử lại."
        except Exception as e:
            logger.error(f"LLM call exception: {e}", exc_info=True)
            reply = "Hệ thống đang bận, vui lòng thử lại sau ít phút."

        return json_response({"ok": True, "reply": reply}, 200)

    except Exception as e:
        logger.error(f"/api/chat unexpected error: {e}", exc_info=True)
        return json_response({"ok": False, "error": "Server error"}, 500)

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        # Return last 30 messages
        with bot._lock:
            msgs = bot.messages[-30:]
        return json_response({"ok": True, "messages": msgs}, 200)
    except Exception as e:
        logger.error(f"API history error: {e}", exc_info=True)
        return json_response({"ok": True, "messages": []}, 200)

@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        bot.clear_history()
        return json_response({"ok": True, "message": "Cleared"}, 200)
    except Exception as e:
        logger.error(f"API clear error: {e}", exc_info=True)
        return json_response({"ok": False, "error": "Error"}, 500)

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        t = bot.get_vietnam_time_info()
        return json_response({"ok": True, "status":"online", "messages_count": len(bot.messages), "vietnam_time": t['current_time']}, 200)
    except Exception as e:
        logger.error(f"API status error: {e}", exc_info=True)
        return json_response({"ok": True, "status":"online"}, 200)

@app.errorhandler(404)
def not_found(e):
    return json_response({"ok": False, "error": "Not found"}, 404)

@app.errorhandler(500)
def internal_err(e):
    return json_response({"ok": False, "error": "Server error"}, 500)

# Optional: small background saver stub (no-op now but ready)
@async_task
def background_maintainer():
    while True:
        try:
            # future: persist messages to disk/db if needed
            time.sleep(5)
        except Exception:
            time.sleep(1)

if __name__ == "__main__":
    # Start background thread
    executor.submit(background_maintainer)
    port = int(os.getenv("PORT", "5000"))
    # debug=False for production; threaded=True to handle multiple requests
    logger.info(f"Starting server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
