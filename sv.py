import os
import json
import time
import requests
import pytz
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS

# Lazy import g4f để tránh lỗi khi Vercel build
try:
    from g4f.client import Client
    from g4f.errors import G4fError
    G4F_AVAILABLE = True
except ImportError:
    G4F_AVAILABLE = False

# --- Cấu hình ---
# Lấy cấu hình từ biến môi trường của Vercel
VIETNAM_TZ = pytz.timezone(os.getenv("VIETNAM_TZ", "Asia/Ho_Chi_Minh"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_CALL_TIMEOUT = int(os.getenv("API_CALL_TIMEOUT", 25)) # Tăng timeout cho g4f
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20)) # Số lượng tin nhắn tối đa trong lịch sử
MAX_PAYLOAD_MESSAGES = int(os.getenv("MAX_PAYLOAD_MESSAGES", 5)) # Số tin nhắn gần nhất gửi cho AI

# URL API để lưu trữ lịch sử chat. BẮT BUỘC PHẢI CÓ.
# Ví dụ: JSONBin.io, npoint.io, hoặc một API riêng của bạn.
STORAGE_API_URL = os.getenv("STORAGE_API_URL")
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY") # Một số dịch vụ yêu cầu API key

# --- Khởi tạo Flask App ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.logger.disabled = True # Tắt logger mặc định của Flask

# Khởi tạo g4f client (nếu có)
if G4F_AVAILABLE:
    client = Client()

# --- Các hàm tiện ích ---

def get_vietnam_time_info():
    """Lấy thông tin thời gian hiện tại ở Việt Nam."""
    now = datetime.now(VIETNAM_TZ)
    weekdays = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    months = [f'tháng {i}' for i in range(1, 13)]
    
    return {
        'current_time': now.strftime('%H:%M:%S'),
        'full_datetime': f"{weekdays[now.weekday()]}, ngày {now.day} {months[now.month - 1]} năm {now.year}, lúc {now.strftime('%H:%M:%S')}",
        'location': 'Phan Rang–Tháp Chàm, Ninh Thuận, Vietnam' # Cập nhật vị trí hiện tại
    }

def get_storage_headers():
    """Tạo headers để gọi API lưu trữ."""
    headers = {"Content-Type": "application/json"}
    if STORAGE_API_KEY:
        # Nhiều dịch vụ dùng 'X-Master-Key', 'X-Access-Key' hoặc 'Authorization'
        headers["X-Master-Key"] = STORAGE_API_KEY
    return headers

def load_history():
    """Tải lịch sử chat từ dịch vụ lưu trữ ngoài."""
    if not STORAGE_API_URL:
        return []
    try:
        res = requests.get(STORAGE_API_URL, headers=get_storage_headers(), timeout=5)
        if res.status_code == 200:
            data = res.json()
            # Hỗ trợ cả 2 định dạng: {"record": [...]} của JSONBin hoặc [...]
            messages = data.get("record", data) if isinstance(data, dict) else data
            if isinstance(messages, list):
                return messages[-MAX_HISTORY_MESSAGES:]
    except (requests.RequestException, json.JSONDecodeError) as e:
        app.logger.error(f"Error loading history: {e}")
    return []

def save_history(messages):
    """Lưu lịch sử chat ra dịch vụ lưu trữ ngoài."""
    if not STORAGE_API_URL:
        return False
    try:
        # Giới hạn lịch sử trước khi lưu
        limited_messages = messages[-MAX_HISTORY_MESSAGES:]
        # JSONBin.io yêu cầu gói dữ liệu trong một key, ví dụ "record"
        # Với các dịch vụ khác, bạn có thể chỉ cần gửi `json=limited_messages`
        payload = {"messages": limited_messages} # Dùng một key chung
        
        res = requests.put(STORAGE_API_URL, json=payload, headers=get_storage_headers(), timeout=5)
        return res.status_code == 200
    except requests.RequestException as e:
        app.logger.error(f"Error saving history: {e}")
    return False

def build_prompt(messages, user_input):
    """Xây dựng prompt cho AI từ lịch sử và thông tin hệ thống."""
    time_info = get_vietnam_time_info()
    system_prompt = {
        "role": "system",
        "content": f"You are a helpful AI assistant. Current location: {time_info['location']}. Current time: {time_info['full_datetime']}."
    }
    
    # Lấy những tin nhắn gần nhất để làm ngữ cảnh
    recent_messages = messages[-(MAX_PAYLOAD_MESSAGES - 1):] if messages else []
    
    payload = [system_prompt] + recent_messages + [{"role": "user", "content": user_input}]
    return payload


# --- API Endpoints ---

@app.route("/")
def index():
    return render_template("sv.html")

@app.route("/api/chat", methods=["POST"])
def api_chat_stream():
    """Endpoint chính, sử dụng streaming để trả lời."""
    data = request.get_json(silent=True)
    if not data or not isinstance(data.get("message"), str):
        return jsonify({"error": "Invalid JSON or missing message"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if len(user_message) > 1500:
        return jsonify({"error": "Message too long"}), 400

    if user_message.lower() in ("clear", "/clear", "xóa", "xoa"):
        save_history([])
        return jsonify({"reply": "Đã xóa lịch sử trò chuyện."})

    if not G4F_AVAILABLE:
         return jsonify({"error": "g4f library is not available."}), 500

    def generate_response():
        # 1. Tải lịch sử cũ
        messages = load_history()
        
        # 2. Tạo payload cho AI
        payload = build_prompt(messages, user_message)
        
        full_response = ""
        try:
            # 3. Gọi g4f ở chế độ stream
            response_stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=payload,
                stream=True,
                timeout=API_CALL_TIMEOUT,
            )
            
            # 4. Gửi từng phần của phản hồi về client
            for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    # Dùng Server-Sent Events (SSE) format
                    yield f"data: {json.dumps({'delta': content})}\n\n"

        except G4fError as e:
            app.logger.error(f"g4f Error: {e}")
            error_message = "Xin lỗi, có lỗi xảy ra từ phía AI. Vui lòng thử lại sau."
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            full_response = error_message # Lưu lỗi vào history
        except Exception as e:
            app.logger.error(f"Generic Error: {e}")
            error_message = "Hệ thống đang bận, vui lòng thử lại sau ít phút."
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            full_response = error_message # Lưu lỗi vào history

        # 5. Sau khi stream kết thúc, cập nhật và lưu lại toàn bộ lịch sử
        if full_response:
             messages.append({"role": "user", "content": user_message})
             messages.append({"role": "assistant", "content": full_response.strip()})
             save_history(messages)

    # Trả về một Response object với kiểu content là text/event-stream
    return Response(generate_response(), mimetype='text/event-stream')


@app.route("/api/history", methods=["GET"])
def api_history():
    messages = load_history()
    return jsonify({"messages": messages})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    if save_history([]):
        return jsonify({"message": "History cleared"})
    return jsonify({"error": "Failed to clear history"}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    time_info = get_vietnam_time_info()
    return jsonify({
        "status": "online", 
        "model": MODEL_NAME,
        "vietnam_time": time_info['current_time']
    })

# Bắt lỗi chung
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error"}), 500

# Dòng này chỉ để chạy local, Vercel sẽ không dùng đến
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
