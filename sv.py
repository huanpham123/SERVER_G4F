from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import tempfile
import threading
import time
import asyncio
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from g4f.client import Client
from jinja2 import TemplateNotFound
import logging
from functools import lru_cache, wraps
import pytz

# Khởi tạo timezone Khánh Hòa, Việt Nam (UTC+7)
VIETNAM_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# Configure lightweight logging
logging.basicConfig(
    level=logging.WARNING,  # Chỉ log warnings và errors
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Ultra-Optimized Configuration ---
HISTORY_FILE = os.environ.get("HISTORY_FILE", "chat_history.json")
MAX_HISTORY_FILE_BYTES = 3 * 1024 * 1024  # 3MB for faster I/O
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", 80))  # Reduced for speed
MAX_INPUT_MESSAGES = int(os.environ.get("MAX_INPUT_MESSAGES", 3))  # Minimal context
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_CALL_TIMEOUT = float(os.environ.get("API_CALL_TIMEOUT", 15.0))  # Aggressive timeout

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chatbot")

# Flask app with ultra optimizations
app = Flask(__name__, 
           static_folder="static", 
           template_folder="templates",
           instance_relative_config=True)

# Optimized CORS
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Disable all Flask logging for maximum performance
app.logger.disabled = True
logging.getLogger('werkzeug').setLevel(logging.ERROR)

def async_task(func):
    """Decorator to run function in thread pool"""
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
    
    def __init__(self, history_file=HISTORY_FILE, max_messages=MAX_MESSAGES):
        if hasattr(self, 'initialized'):
            return
            
        self.client = Client()
        self.history_file = history_file
        self.max_messages = max_messages
        self.messages = []
        self._message_cache = {}
        self._last_save_time = 0
        self._save_interval = 3.0  # Longer interval for less I/O
        self._lock = threading.RLock()
        
        # Pre-load timezone info
        self._vietnam_tz = VIETNAM_TZ
        
        self._ensure_history_size()
        self._load_history()
        self.initialized = True

    def get_vietnam_time_info(self):
        """Lấy thông tin thời gian chi tiết cho Khánh Hòa, Việt Nam"""
        vn_time = datetime.now(self._vietnam_tz)
        
        # Tiếng Việt weekdays và months
        weekdays = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
        months = ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6',
                 'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12']
        
        weekday_vn = weekdays[vn_time.weekday()]
        month_vn = months[vn_time.month - 1]
        
        time_info = {
            'current_time': vn_time.strftime('%H:%M:%S'),
            'current_date': f"{vn_time.day} {month_vn} năm {vn_time.year}",
            'weekday': weekday_vn,
            'full_datetime': f"{weekday_vn}, {vn_time.day} {month_vn} {vn_time.year} lúc {vn_time.strftime('%H:%M:%S')}",
            'timezone': 'UTC+7 (Giờ Việt Nam)',
            'location': 'Khánh Hòa, Việt Nam',
            'timestamp': vn_time.timestamp()
        }
        return time_info

    def _ensure_history_size(self):
        """Ultra-fast file size check"""
        try:
            if os.path.exists(self.history_file):
                if os.path.getsize(self.history_file) > MAX_HISTORY_FILE_BYTES:
                    os.remove(self.history_file)
                    self.messages = []
        except (OSError, IOError):
            pass

    def _load_history(self):
        """Lightning-fast history loading"""
        if not os.path.exists(self.history_file):
            self.messages = []
            return
            
        try:
            with open(self.history_file, "r", encoding="utf-8", buffering=16384) as f:
                content = f.read()
                if content.strip():
                    self.messages = json.loads(content)
                    # Aggressive trimming on load
                    if len(self.messages) > self.max_messages:
                        self.messages = self.messages[-self.max_messages:]
                else:
                    self.messages = []
        except:
            self.messages = []

    def _atomic_save(self, path, data):
        """Ultra-fast atomic save with minimal JSON"""
        dir_path = os.path.dirname(path) or "."
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                encoding='utf-8', 
                dir=dir_path, 
                delete=False,
                buffering=16384
            ) as tf:
                # Minimal JSON without indentation
                json.dump(data, tf, ensure_ascii=False, separators=(',', ':'))
                tmpname = tf.name
            os.replace(tmpname, path)
        except:
            if 'tmpname' in locals() and os.path.exists(tmpname):
                try:
                    os.unlink(tmpname)
                except:
                    pass

    @async_task
    def save_history_async(self, force=False):
        """Async non-blocking save"""
        current_time = time.time()
        if not force and (current_time - self._last_save_time) < self._save_interval:
            return
            
        with self._lock:
            try:
                # Aggressive message trimming
                if len(self.messages) > self.max_messages:
                    system_msgs = [m for m in self.messages if m.get("role") == "system"]
                    recent_msgs = [m for m in self.messages if m.get("role") != "system"][-50:]
                    self.messages = system_msgs + recent_msgs
                
                self._atomic_save(self.history_file, self.messages)
                self._last_save_time = current_time
            except:
                pass

    def manage_memory_aggressive(self):
        """Aggressive memory management"""
        if len(self.messages) > self.max_messages:
            with self._lock:
                system_msgs = [m for m in self.messages if m.get("role") == "system"]
                recent = [m for m in self.messages if m.get("role") != "system"][-40:]
                self.messages = system_msgs + recent
                self._message_cache.clear()

    @lru_cache(maxsize=1)
    def _get_enhanced_system_prompt(self):
        """Cached system prompt với thông tin thời gian Việt Nam"""
        time_info = self.get_vietnam_time_info()
        
        prompt = f"""Bạn là một trợ lý AI thông minh và hữu ích, hiện đang ở Khánh Hòa, Việt Nam.

THÔNG TIN THỜI GIAN HIỆN TẠI:
- Thời gian: {time_info['current_time']} ({time_info['timezone']})
- Ngày: {time_info['full_datetime']}
- Địa điểm: {time_info['location']}

Hãy trả lời ngắn gọn, chính xác và thân thiện. Khi được hỏi về thời gian, hãy sử dụng thông tin thời gian hiện tại ở trên. Bạn có thể tham khảo ngày giờ để đưa ra lời khuyên phù hợp."""
        
        return prompt

    def add_system_message_with_time(self):
        """Add system message with current Vietnam time"""
        # Clear old system messages
        self.messages = [m for m in self.messages if m.get("role") != "system"]
        
        # Add new system message with current time
        time_info = self.get_vietnam_time_info()
        system_content = self._get_enhanced_system_prompt()
        
        self.messages.insert(0, {
            "role": "system", 
            "content": system_content,
            "timestamp": time_info['timestamp']
        })

    def clear_history(self):
        """Lightning-fast clear"""
        with self._lock:
            self.messages = []
            self._message_cache.clear()
            try:
                if os.path.exists(self.history_file):
                    os.remove(self.history_file)
            except:
                pass

    def _build_minimal_payload(self, user_input):
        """Ultra-minimal payload for fastest API calls"""
        # Get only the most essential messages
        system_msgs = [{"role": m["role"], "content": m["content"]} 
                      for m in self.messages if m.get("role") == "system"]
        
        # Only last 2-3 exchanges
        non_system = [{"role": m["role"], "content": m["content"]} 
                     for m in self.messages if m.get("role") in ("user", "assistant")]
        
        recent_msgs = non_system[-MAX_INPUT_MESSAGES:] if non_system else []
        
        # Minimal payload
        payload = system_msgs + recent_msgs + [{"role": "user", "content": user_input}]
        return payload

    def get_response_ultra_fast(self, user_input):
        """Ultra-optimized response with Vietnam timezone"""
        start_time = time.time()
        
        try:
            # Update system message with current time every few requests
            if len(self.messages) % 5 == 0:  # Every 5 messages
                self.add_system_message_with_time()
            elif not any(m.get("role") == "system" for m in self.messages):
                self.add_system_message_with_time()
            
            # Build ultra-minimal payload
            payload_messages = self._build_minimal_payload(user_input)
            
            # Ultra-aggressive API call settings
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=payload_messages,
                max_tokens=800,  # Reduced for speed
                temperature=0.5,  # Reduced for consistency
                top_p=0.9
            )
            
            # Extract response with fallbacks
            try:
                bot_reply = response.choices[0].message.content.strip()
            except:
                try:
                    bot_reply = str(response).strip()
                except:
                    bot_reply = "Xin lỗi, có lỗi kỹ thuật. Vui lòng thử lại."
            
            if not bot_reply or len(bot_reply) < 2:
                bot_reply = "Tôi chưa hiểu rõ câu hỏi. Bạn có thể hỏi lại được không?"
            
            # Add to history with Vietnam timestamp
            vn_time = datetime.now(self._vietnam_tz)
            timestamp = vn_time.strftime("%Y-%m-%d %H:%M:%S")
            
            self.messages.extend([
                {"role": "user", "content": user_input, "timestamp": timestamp},
                {"role": "assistant", "content": bot_reply, "timestamp": timestamp}
            ])
            
            # Async memory management and save
            if len(self.messages) > self.max_messages * 0.8:
                self.manage_memory_aggressive()
            
            # Non-blocking save
            self.save_history_async()
            
            response_time = time.time() - start_time
            if response_time > 10:  # Log slow responses
                app.logger.warning(f"Slow response: {response_time:.2f}s")
            
            return bot_reply
            
        except Exception as e:
            error_msg = "Hệ thống đang bận, vui lòng thử lại sau ít phút."
            
            # Still save user message
            try:
                vn_time = datetime.now(self._vietnam_tz)
                self.messages.append({
                    "role": "user", 
                    "content": user_input, 
                    "timestamp": vn_time.strftime("%Y-%m-%d %H:%M:%S")
                })
            except:
                pass
                
            return error_msg

# Global bot instance
bot = UltraFastChatBot()

# Ultra-optimized Routes
@app.route("/")
def index():
    try:
        return render_template("sv.html")
    except TemplateNotFound:
        return """
        <h2>404 — templates/sv.html not found</h2>
        <p>Please place <b>sv.html</b> in <code>templates/</code> folder.</p>
        """, 404

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Ultra-fast chat endpoint"""
    try:
        # Ultra-fast validation
        if not request.is_json:
            return jsonify({"ok": False, "error": "Invalid content type"}), 400
            
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"ok": False, "error": "Invalid JSON"}), 400
            
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"ok": False, "error": "Empty message"}), 400
            
        if len(message) > 1500:  # Stricter limit
            return jsonify({"ok": False, "error": "Message too long"}), 400
        
        # Quick commands
        if message.lower() in ("clear", "/clear", "xóa", "xoa"):
            bot.clear_history()
            return jsonify({"ok": True, "reply": "Đã xóa lịch sử chat."})
        
        # Time query check
        time_keywords = ["giờ", "thời gian", "ngày", "tháng", "năm", "bây giờ", "hiện tại"]
        if any(keyword in message.lower() for keyword in time_keywords):
            time_info = bot.get_vietnam_time_info()
            time_reply = f"Hiện tại là {time_info['full_datetime']} tại {time_info['location']}."
            
            # Add to history
            vn_time = datetime.now(bot._vietnam_tz)
            timestamp = vn_time.strftime("%Y-%m-%d %H:%M:%S")
            bot.messages.extend([
                {"role": "user", "content": message, "timestamp": timestamp},
                {"role": "assistant", "content": time_reply, "timestamp": timestamp}
            ])
            bot.save_history_async()
            
            return jsonify({"ok": True, "reply": time_reply})
        
        # Get AI response
        reply = bot.get_response_ultra_fast(message)
        return jsonify({"ok": True, "reply": reply})
        
    except Exception as e:
        return jsonify({"ok": False, "error": "Server busy"}), 500

@app.route("/api/time", methods=["GET"])
def api_time():
    """Fast time endpoint for Vietnam timezone"""
    try:
        time_info = bot.get_vietnam_time_info()
        return jsonify({"ok": True, "time_info": time_info})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    """Ultra-fast history endpoint"""
    try:
        # Only return last 30 messages for speed
        recent_messages = bot.messages[-30:] if bot.messages else []
        return jsonify({"ok": True, "messages": recent_messages})
    except:
        return jsonify({"ok": True, "messages": []})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Lightning-fast clear"""
    try:
        bot.clear_history()
        return jsonify({"ok": True, "message": "Cleared"})
    except:
        return jsonify({"ok": False, "error": "Error"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    """Ultra-fast status check"""
    try:
        time_info = bot.get_vietnam_time_info()
        return jsonify({
            "ok": True, 
            "status": "online",
            "messages_count": len(bot.messages),
            "model": MODEL_NAME,
            "vietnam_time": time_info['current_time'],
            "location": "Khánh Hòa, Việt Nam"
        })
    except:
        return jsonify({"ok": True, "status": "online"})

# Minimal error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"ok": False, "error": "Server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    # Ultra-high performance settings
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=False, 
        threaded=True,
        use_reloader=False,
        use_debugger=False,
        passthrough_errors=False
    )