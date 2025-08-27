# app.py
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pytz
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ---------- Config ----------
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "100"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "6.0"))    # seconds for LLM calls
STORAGE_API_URL = os.getenv("STORAGE_API_URL", "")     # optional external storage endpoint
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY", "")
VIETNAM_TZ = pytz.timezone(os.getenv("VIETNAM_TZ", "Asia/Ho_Chi_Minh"))

# Thread pool for non-blocking saves
executor = ThreadPoolExecutor(max_workers=2)

# Flask app
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------- Simple in-memory storage ----------
class VercelChatBot:
    def __init__(self):
        self.messages = []  # list of dicts: {"role": "user"/"assistant", "content": "...", "timestamp": "..."}
        self._last_request_time = 0

    def _now_ts(self):
        return datetime.now(VIETNAM_TZ).isoformat()

    def get_vietnam_time(self):
        vn = datetime.now(VIETNAM_TZ)
        return {
            "time": vn.strftime("%H:%M:%S"),
            "date": vn.strftime("%d/%m/%Y"),
            "weekday": ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"][vn.weekday()],
            "full": f"{vn.strftime('%H:%M:%S')} - {vn.strftime('%d/%m/%Y')}",
            "location": "Khánh Hòa, Việt Nam"
        }

    def append_messages(self, role, content):
        try:
            self.messages.append({"role": role, "content": content, "timestamp": self._now_ts()})
            # trim
            if len(self.messages) > MAX_MESSAGES:
                self.messages = self.messages[-MAX_MESSAGES:]
        except Exception:
            pass

    def save_history_local(self):
        """Try to save locally into a tmp file. On Vercel this is ephemeral but may help during single run."""
        try:
            path = "/tmp/chat_history.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.messages[-MAX_MESSAGES:], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def save_history_external(self):
        """Try to POST history to an external STORAGE_API_URL (non-blocking)."""
        if not STORAGE_API_URL:
            return

        def _post():
            try:
                headers = {"Content-Type": "application/json"}
                if STORAGE_API_KEY:
                    headers["Authorization"] = f"Bearer {STORAGE_API_KEY}"
                # short timeout to avoid blocking
                requests.post(
                    f"{STORAGE_API_URL.rstrip('/')}/history",
                    json={"messages": self.messages[-MAX_MESSAGES:]},
                    headers=headers,
                    timeout=2
                )
            except Exception:
                pass

        try:
            executor.submit(_post)
        except Exception:
            pass

    def save_history(self):
        # local (fast)
        self.save_history_local()
        # external (best effort, non-blocking)
        self.save_history_external()

    def get_response_fast(self, user_input):
        # Rate-limit quick protection
        if time.time() - self._last_request_time < 0.3:
            return "Vui lòng chờ một chút trước khi gửi tin nhắn tiếp theo."

        self._last_request_time = time.time()

        # fast handling for time queries
        time_keywords = ["giờ", "thời gian", "ngày", "tháng", "năm", "bây giờ", "hiện tại"]
        if any(k in user_input.lower() for k in time_keywords):
            ti = self.get_vietnam_time()
            reply = f"Hiện tại là {ti['full']} tại {ti['location']}."
            self.append_messages("user", user_input)
            self.append_messages("assistant", reply)
            # save asynchronously
            threading.Thread(target=self.save_history, daemon=True).start()
            return reply

        # Build minimal context (last 3 messages)
        try:
            recent = [m for m in self.messages if m.get("role") in ("user","assistant")][-3:]
        except Exception:
            recent = []

        # Try to call g4f quickly (if available), otherwise fallback
        bot_reply = None
        try:
            # Import lazily so deploy works even if g4f not installed
            from g4f.client import Client
            client = Client()
            api_messages = [{"role":"system","content":f"You are an assistant in {self.get_vietnam_time()['location']}. Answer short and helpful."}]
            api_messages += [{"role": m["role"], "content": m["content"]} for m in recent]
            api_messages.append({"role":"user","content": user_input})

            # call with a timeout; if it raises/blocks, we fallback
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=api_messages,
                max_tokens=400,
                temperature=0.7,
                timeout=API_TIMEOUT
            )
            # The structure may vary depending on g4f version
            try:
                bot_reply = resp.choices[0].message.content.strip()
            except Exception:
                # fallback to string if direct
                bot_reply = str(resp).strip()[:2000]
        except Exception:
            # any error -> quick fallback
            bot_reply = None

        if not bot_reply or len(bot_reply) < 3:
            # fallback quick answer
            bot_reply = "Xin lỗi, hiện tại tôi không truy cập được mô-đun trả lời nhanh. Bạn có thể thử lại hoặc chờ một chút."

        # append and save (best-effort, non-blocking)
        self.append_messages("user", user_input)
        self.append_messages("assistant", bot_reply)
        threading.Thread(target=self.save_history, daemon=True).start()

        return bot_reply

    def clear_history(self):
        self.messages = []
        # try clear local file
        try:
            path = "/tmp/chat_history.json"
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        # optionally clear external
        if STORAGE_API_URL:
            def _del():
                try:
                    headers = {}
                    if STORAGE_API_KEY:
                        headers["Authorization"] = f"Bearer {STORAGE_API_KEY}"
                    requests.delete(f"{STORAGE_API_URL.rstrip('/')}/history", headers=headers, timeout=2)
                except Exception:
                    pass
            try:
                executor.submit(_del)
            except Exception:
                pass

bot = VercelChatBot()

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("sv.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(force=True)
        if not data or "message" not in data:
            return jsonify({"ok": False, "error": "Invalid request"}), 400

        message = data["message"].strip()
        if not message or len(message) > 5000:
            return jsonify({"ok": False, "error": "Invalid message length"}), 400

        # handle quick clear keywords
        if message.lower() in ["clear","/clear","xóa","xoa"]:
            bot.clear_history()
            return jsonify({"ok": True, "reply": "Đã xóa lịch sử chat."})

        # get reply (fast)
        reply = bot.get_response_fast(message)
        return jsonify({"ok": True, "reply": reply})
    except Exception:
        return jsonify({"ok": False, "error": "Server busy"}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        # return last 50 messages
        return jsonify({"ok": True, "messages": bot.messages[-50:]})
    except Exception:
        return jsonify({"ok": True, "messages": []})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        bot.clear_history()
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False, "error": "Error"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        t = bot.get_vietnam_time()
        return jsonify({"ok": True, "status": "online", "time": t["full"], "messages": len(bot.messages)})
    except Exception:
        return jsonify({"ok": True, "status": "online"})

@app.route("/api/time", methods=["GET"])
def api_time():
    try:
        t = bot.get_vietnam_time()
        return jsonify({"ok": True, "time_info": t})
    except Exception:
        return jsonify({"ok": False, "error": "Time error"}), 500

# For local dev
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
