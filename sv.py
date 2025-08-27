import os
import json
import requests
import pytz
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS

# Lazy import g4f để Vercel build không bị lỗi
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
API_CALL_TIMEOUT = int(os.getenv("API_CALL_TIMEOUT", 25))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20))
MAX_PAYLOAD_MESSAGES = int(os.getenv("MAX_PAYLOAD_MESSAGES", 5))

# URL API để lưu trữ lịch sử chat. BẮT BUỘC PHẢI CÓ ĐỂ DEPLOY.
STORAGE_API_URL = os.getenv("STORAGE_API_URL")
# Một số dịch vụ (như JSONBin.io) yêu cầu API key.
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY")

# --- Khởi tạo Flask App ---
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Khởi tạo g4f client (nếu có)
client = Client() if G4F_AVAILABLE else None

# --- Các hàm tiện ích ---

def get_vietnam_time_info():
    """Lấy thông tin thời gian hiện tại ở Việt Nam."""
    now = datetime.now(VIETNAM_TZ)
    return {
        'full_datetime': now.strftime('%A, %d/%m/%Y, %H:%M:%S'),
        'location': 'Phan Rang–Tháp Chàm, Ninh Thuận, Vietnam'
    }

def get_storage_headers():
    """Tạo headers để gọi API lưu trữ."""
    headers = {"Content-Type": "application/json"}
    if STORAGE_API_KEY:
        # JSONBin.io dùng 'X-Master-Key'. Dịch vụ khác có thể dùng key khác.
        headers["X-Master-Key"] = STORAGE_API_KEY
    return headers

def load_history():
    """Tải lịch sử chat từ dịch vụ lưu trữ ngoài."""
    if not STORAGE_API_URL:
        return []
    try:
        res = requests.get(f"{STORAGE_API_URL}/latest", headers=get_storage_headers(), timeout=5)
        if res.status_code == 200:
            data = res.json()
            # JSONBin.io gói dữ liệu trong key "record".
            messages = data.get("record", [])
            return messages[-MAX_HISTORY_MESSAGES:]
    except (requests.RequestException, json.JSONDecodeError):
        pass # Bỏ qua lỗi nếu không tải được
    return []

def save_history(messages):
    """Lưu lịch sử chat ra dịch vụ lưu trữ ngoài."""
    if not STORAGE_API_URL:
        return False
    try:
        limited_messages = messages[-MAX_HISTORY_MESSAGES:]
        res = requests.put(STORAGE_API_URL, json=limited_messages, headers=get_storage_headers(), timeout=5)
        return res.status_code == 200
    except requests.RequestException:
        return False

def build_prompt(messages, user_input):
    """Xây dựng prompt cho AI từ lịch sử và thông tin hệ thống."""
    time_info = get_vietnam_time_info()
    system_prompt = {
        "role": "system",
        "content": f"You are a helpful AI assistant. Current location: {time_info['location']}. Current time: {time_info['full_datetime']}."
    }
    recent_messages = messages[-(MAX_PAYLOAD_MESSAGES - 1):] if messages else []
    return [system_prompt] + recent_messages + [{"role": "user", "content": user_input}]

# --- API Endpoints ---

@app.route("/")
def home():
    return render_template("sv.html")

@app.route("/api/chat", methods=["POST"])
def api_chat_stream():
    """Endpoint chính, sử dụng streaming để trả lời, chống timeout."""
    data = request.get_json(silent=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if not client:
        return jsonify({"error": "g4f library is not available."}), 500

    def generate_response():
        messages = load_history()
        payload = build_prompt(messages, user_message)
        full_response = ""
        
        try:
            response_stream = client.chat.completions.create(
                model=MODEL_NAME, messages=payload, stream=True, timeout=API_CALL_TIMEOUT
            )
            for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield f"data: {json.dumps({'delta': content})}\n\n"
        
        except Exception as e:
            error_message = f"Lỗi: {str(e)}"
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            full_response = error_message
        
        # Sau khi stream kết thúc, cập nhật và lưu lại lịch sử
        if full_response:
             messages.append({"role": "user", "content": user_message})
             messages.append({"role": "assistant", "content": full_response.strip()})
             save_history(messages)

    return Response(generate_response(), mimetype='text/event-stream')

@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({"messages": load_history()})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    if save_history([]):
        return jsonify({"message": "History cleared"})
    return jsonify({"error": "Failed to clear history"}), 500
