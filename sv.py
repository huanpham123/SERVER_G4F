from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import requests
import asyncio
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pytz
from functools import lru_cache
import time
import logging

# Disable all logging for maximum performance
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

# Vietnam timezone
VIETNAM_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# Configuration for Vercel
MAX_MESSAGES = 50  # Reduced for Vercel memory limits
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_TIMEOUT = 10.0  # Aggressive timeout for Vercel

# External storage URL for chat history (use your own)
STORAGE_API_URL = os.environ.get("STORAGE_API_URL", "")
STORAGE_API_KEY = os.environ.get("STORAGE_API_KEY", "")

# Thread pool with minimal workers for Vercel
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat")

# Ultra-lightweight Flask app
app = Flask(__name__)
app.logger.disabled = True

# Minimal CORS
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

class VercelChatBot:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.messages = []
        self._cache = {}
        self._last_request_time = 0
        self._vietnam_tz = VIETNAM_TZ
        
        # Load history from external storage if available
        self._load_external_history()
        self.initialized = True

    @lru_cache(maxsize=1)
    def get_vietnam_time(self):
        """Get current Vietnam time - cached for performance"""
        vn_time = datetime.now(self._vietnam_tz)
        return {
            'time': vn_time.strftime('%H:%M:%S'),
            'date': vn_time.strftime('%d/%m/%Y'),
            'weekday': ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật'][vn_time.weekday()],
            'full': f"{vn_time.strftime('%H:%M:%S')} - {vn_time.strftime('%d/%m/%Y')}",
            'location': 'Khánh Hòa, Việt Nam'
        }

    def _load_external_history(self):
        """Load chat history from external storage"""
        if not STORAGE_API_URL:
            return
            
        try:
            response = requests.get(
                f"{STORAGE_API_URL}/history",
                headers={"Authorization": f"Bearer {STORAGE_API_KEY}"},
                timeout=3
            )
            if response.status_code == 200:
                data = response.json()
                self.messages = data.get('messages', [])[-MAX_MESSAGES:]
        except:
            pass

    def _save_external_history(self):
        """Save to external storage (non-blocking)"""
        if not STORAGE_API_URL:
            return
            
        def save_async():
            try:
                requests.post(
                    f"{STORAGE_API_URL}/history",
                    json={"messages": self.messages[-MAX_MESSAGES:]},
                    headers={"Authorization": f"Bearer {STORAGE_API_KEY}"},
                    timeout=2
                )
            except:
                pass
        
        executor.submit(save_async)

    def get_system_prompt(self):
        """Dynamic system prompt with Vietnam time"""
        time_info = self.get_vietnam_time()
        return f"""Bạn là trợ lý AI thông minh và hữu ích ở Khánh Hòa, Việt Nam.

THÔNG TIN THỜI GIAN HIỆN TẠI:
- Thời gian: {time_info['time']} (UTC+7)
- Ngày: {time_info['date']}
- Thứ: {time_info['weekday']}
- Địa điểm: {time_info['location']}

Hãy trả lời ngắn gọn, chính xác và thân thiện. Khi được hỏi về thời gian, sử dụng thông tin trên."""

    def manage_memory(self):
        """Aggressive memory management for Vercel"""
        if len(self.messages) > MAX_MESSAGES:
            # Keep only system messages and recent conversations
            system_msgs = [m for m in self.messages if m.get("role") == "system"]
            recent_msgs = [m for m in self.messages if m.get("role") != "system"][-30:]
            self.messages = system_msgs + recent_msgs
            self._cache.clear()

    def get_response_fast(self, user_input):
        """Ultra-fast response optimized for Vercel"""
        start_time = time.time()
        
        # Rate limiting
        if time.time() - self._last_request_time < 0.5:
            return "Vui lòng chờ một chút trước khi gửi tin nhắn tiếp theo."
        
        self._last_request_time = time.time()
        
        try:
            # Check for time-related queries first
            time_keywords = ["giờ", "thời gian", "ngày", "tháng", "năm", "bây giờ", "hiện tại"]
            if any(keyword in user_input.lower() for keyword in time_keywords):
                time_info = self.get_vietnam_time()
                reply = f"Hiện tại là {time_info['full']} tại {time_info['location']}."
                
                # Add to history
                timestamp = datetime.now(self._vietnam_tz).isoformat()
                self.messages.extend([
                    {"role": "user", "content": user_input, "timestamp": timestamp},
                    {"role": "assistant", "content": reply, "timestamp": timestamp}
                ])
                
                self._save_external_history()
                return reply

            # Build minimal context for API call
            system_prompt = self.get_system_prompt()
            
            # Only use last 3 messages for context to minimize API call size
            recent_context = []
            if self.messages:
                recent_msgs = [m for m in self.messages if m.get("role") in ["user", "assistant"]][-3:]
                recent_context = [{"role": m["role"], "content": m["content"]} for m in recent_msgs]
            
            # Construct minimal API payload
            api_messages = [{"role": "system", "content": system_prompt}] + recent_context + [{"role": "user", "content": user_input}]
            
            # Use g4f for free API calls
            try:
                from g4f.client import Client
                client = Client()
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    max_tokens=500,  # Reduced for speed
                    temperature=0.7,
                    timeout=API_TIMEOUT
                )
                
                bot_reply = response.choices[0].message.content.strip()
                
            except Exception as api_error:
                # Fallback to simple response if API fails
                bot_reply = "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
            
            if not bot_reply or len(bot_reply) < 3:
                bot_reply = "Tôi chưa hiểu rõ câu hỏi. Bạn có thể hỏi lại được không?"
            
            # Add to history with timestamp
            timestamp = datetime.now(self._vietnam_tz).isoformat()
            self.messages.extend([
                {"role": "user", "content": user_input, "timestamp": timestamp},
                {"role": "assistant", "content": bot_reply, "timestamp": timestamp}
            ])
            
            # Memory management
            self.manage_memory()
            
            # Save to external storage (non-blocking)
            self._save_external_history()
            
            # Log slow responses
            response_time = time.time() - start_time
            if response_time > 8:
                print(f"Slow response: {response_time:.2f}s")
            
            return bot_reply
            
        except Exception as e:
            # Always save user message even if response fails
            try:
                timestamp = datetime.now(self._vietnam_tz).isoformat()
                self.messages.append({
                    "role": "user", 
                    "content": user_input, 
                    "timestamp": timestamp
                })
            except:
                pass
            
            return "Hệ thống đang bận, vui lòng thử lại sau."

    def clear_history(self):
        """Clear all chat history"""
        self.messages = []
        self._cache.clear()
        
        # Clear external storage too
        if STORAGE_API_URL:
            def clear_async():
                try:
                    requests.delete(
                        f"{STORAGE_API_URL}/history",
                        headers={"Authorization": f"Bearer {STORAGE_API_KEY}"},
                        timeout=2
                    )
                except:
                    pass
            
            executor.submit(clear_async)

# Global bot instance
bot = VercelChatBot()

# API Routes optimized for Vercel

@app.route("/")
def index():
    """Serve the main HTML page"""
    html_content = '''<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>⚡ Vercel Chatbot</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh; display: flex; align-items: center; justify-content: center;
      padding: 16px;
    }
    .container { 
      width: 100%; max-width: 800px; 
      background: rgba(255,255,255,0.95); backdrop-filter: blur(10px);
      border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
      padding: 24px; border: 1px solid rgba(255,255,255,0.2);
    }
    .header { text-align: center; margin-bottom: 20px; }
    .header h1 { color: #2d3748; font-size: 24px; font-weight: 700; margin-bottom: 8px; }
    .status { 
      display: inline-block; padding: 4px 12px; border-radius: 12px; 
      font-size: 12px; font-weight: 600; background: #c6f6d5; color: #22543d;
    }
    .chat { 
      height: 400px; overflow-y: auto; border: 1px solid #e2e8f0; 
      border-radius: 16px; padding: 16px; margin-bottom: 20px;
      background: linear-gradient(to bottom, #f7fafc, #edf2f7);
    }
    .message { margin: 12px 0; animation: fadeIn 0.3s ease-out; }
    .message.user { text-align: right; }
    .message.bot { text-align: left; }
    .bubble { 
      display: inline-block; padding: 12px 16px; border-radius: 18px; 
      max-width: 80%; white-space: pre-wrap; word-wrap: break-word;
    }
    .bubble.user { 
      background: linear-gradient(135deg, #4299e1, #3182ce); color: white; 
      box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
    }
    .bubble.bot { 
      background: white; color: #2d3748; border: 1px solid #e2e8f0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .bubble.typing { background: #f7fafc; border: 1px dashed #cbd5e0; }
    .controls { display: flex; gap: 12px; }
    .input-group { flex: 1; position: relative; }
    input { 
      width: 100%; padding: 14px 50px 14px 16px; border-radius: 25px; 
      border: 2px solid #e2e8f0; outline: none; transition: all 0.2s;
    }
    input:focus { border-color: #4299e1; box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1); }
    .send-btn {
      position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
      background: linear-gradient(135deg, #48bb78, #38a169); border: none; 
      border-radius: 50%; width: 36px; height: 36px; cursor: pointer; 
      color: white; display: flex; align-items: center; justify-content: center;
    }
    .send-btn:hover { transform: translateY(-50%) scale(1.1); }
    .clear-btn { 
      padding: 12px 20px; border-radius: 25px; border: none; cursor: pointer; 
      background: #fed7d7; color: #c53030; font-weight: 600;
    }
    .clear-btn:hover { background: #fc8181; color: white; }
    .typing { animation: pulse 1.5s infinite; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>⚡ Vercel AI Chatbot</h1>
      <div class="status">🟢 Online</div>
    </div>
    
    <div class="chat" id="chat"></div>
    
    <div class="controls">
      <div class="input-group">
        <input id="input" placeholder="Nhập câu hỏi và nhấn Enter..." maxlength="1000">
        <button class="send-btn" id="send">➤</button>
      </div>
      <button class="clear-btn" id="clear">🗑️ Xóa</button>
    </div>
  </div>

  <script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    const clearBtn = document.getElementById('clear');
    
    let isTyping = false;
    
    function addMessage(role, text, isTemp = false) {
      const div = document.createElement('div');
      div.className = `message ${role}`;
      
      const bubble = document.createElement('div');
      bubble.className = `bubble ${role} ${isTemp ? 'typing' : ''}`;
      bubble.textContent = isTemp ? 'Đang xử lý...' : text;
      
      div.appendChild(bubble);
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      
      return bubble;
    }
    
    function removeTyping() {
      const typing = chat.querySelectorAll('.bubble.typing');
      typing.forEach(el => el.parentElement.remove());
    }
    
    async function sendMessage() {
      if (isTyping || !input.value.trim()) return;
      
      const message = input.value.trim();
      input.value = '';
      
      addMessage('user', message);
      
      isTyping = true;
      sendBtn.disabled = true;
      input.disabled = true;
      
      const typingBubble = addMessage('bot', '', true);
      
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        removeTyping();
        
        if (data.ok) {
          addMessage('bot', data.reply);
        } else {
          addMessage('bot', data.error || 'Lỗi không xác định');
        }
        
      } catch (error) {
        removeTyping();
        addMessage('bot', 'Lỗi kết nối. Vui lòng thử lại.');
      }
      
      isTyping = false;
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
    }
    
    async function clearChat() {
      if (!confirm('Xóa toàn bộ lịch sử chat?')) return;
      
      try {
        const response = await fetch('/api/clear', { method: 'POST' });
        const data = await response.json();
        
        if (data.ok) {
          chat.innerHTML = '';
        }
      } catch (error) {
        alert('Lỗi xóa chat');
      }
    }
    
    sendBtn.onclick = sendMessage;
    clearBtn.onclick = clearChat;
    
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    
    // Load history
    async function loadHistory() {
      try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.ok && data.messages) {
          data.messages.forEach(msg => {
            if (msg.role && msg.content) {
              addMessage(msg.role === 'user' ? 'user' : 'bot', msg.content);
            }
          });
        }
      } catch (error) {
        console.warn('Could not load history');
      }
    }
    
    loadHistory();
    input.focus();
  </script>
</body>
</html>'''
    return html_content

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Main chat endpoint - ultra-fast response"""
    try:
        data = request.get_json(force=True)
        if not data or not data.get("message"):
            return jsonify({"ok": False, "error": "Invalid request"}), 400
            
        message = data["message"].strip()
        
        if not message or len(message) > 1000:
            return jsonify({"ok": False, "error": "Invalid message length"}), 400
        
        # Handle clear command
        if message.lower() in ["clear", "/clear", "xóa", "xoa"]:
            bot.clear_history()
            return jsonify({"ok": True, "reply": "Đã xóa lịch sử chat."})
        
        # Get AI response
        reply = bot.get_response_fast(message)
        return jsonify({"ok": True, "reply": reply})
        
    except Exception as e:
        return jsonify({"ok": False, "error": "Server busy"}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    """Get chat history"""
    try:
        # Return only last 20 messages for speed
        recent = bot.messages[-20:] if bot.messages else []
        return jsonify({"ok": True, "messages": recent})
    except:
        return jsonify({"ok": True, "messages": []})

@app.route("/api/clear", methods=["POST"])  
def api_clear():
    """Clear chat history"""
    try:
        bot.clear_history()
        return jsonify({"ok": True})
    except:
        return jsonify({"ok": False, "error": "Error"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    """Server status"""
    try:
        time_info = bot.get_vietnam_time()
        return jsonify({
            "ok": True,
            "status": "online", 
            "time": time_info['full'],
            "location": time_info['location'],
            "messages": len(bot.messages)
        })
    except:
        return jsonify({"ok": True, "status": "online"})

@app.route("/api/time", methods=["GET"])
def api_time():
    """Vietnam time endpoint"""
    try:
        time_info = bot.get_vietnam_time()
        return jsonify({"ok": True, "time_info": time_info})
    except:
        return jsonify({"ok": False, "error": "Time error"}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"ok": False, "error": "Server error"}), 500

# Vercel entry point
if __name__ == "__main__":
    app.run(debug=False)
