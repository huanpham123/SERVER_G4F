import os
import json
import requests
import pytz
import logging
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS

# --- Cấu hình logging để dễ dàng debug trên Vercel ---
logging.basicConfig(level=logging.INFO)

# --- Lazy import g4f để Vercel build không bị lỗi ---
try:
    from g4f.client import Client
    # Chọn một vài provider được đánh giá là ổn định hơn
    from g4f.Provider import (
        Liaobots,
        GeekGpt,
        AiChatOnline,
    )
    G4F_AVAILABLE = True
except ImportError as e:
    logging.error(f"g4f import error: {e}")
    G4F_AVAILABLE = False

# --- Cấu hình ứng dụng ---
# Lấy cấu hình từ biến môi trường của Vercel
VIETNAM_TZ = pytz.timezone(os.getenv("VIETNAM_TZ", "Asia/Ho_Chi_Minh"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_CALL_TIMEOUT = int(os.getenv("API_CALL_TIMEOUT", 30)) # Tăng timeout lên một chút
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20))
MAX_PAYLOAD_MESSAGES = int(os.getenv("MAX_PAYLOAD_MESSAGES", 7)) # Gửi nhiều ngữ cảnh hơn một chút

# URL API để lưu trữ lịch sử chat (BẮT BUỘC)
STORAGE_API_URL = os.getenv("STORAGE_API_URL")
STORAGE_API_KEY = os.getenv("STORAGE_API_KEY")

# --- MỚI: Cho phép chọn Provider qua biến môi trường ---
# Bạn có thể đặt G4F_PROVIDER trên Vercel là "GeekGpt", "Liaobots", etc.
# Nếu không đặt, nó sẽ thử Liaobots trước.
DEFAULT_PROVIDER = Liaobots
PROVIDER_MAP = {
    "Liaobots": Liaobots,
    "GeekGpt": GeekGpt,
    "AiChatOnline": AiChatOnline,
}
SELECTED_PROVIDER_NAME = os.getenv("G4F_PROVIDER", "Liaobots")
G4F_PROVIDER = PROVIDER_MAP.get(SELECTED_PROVIDER_NAME, DEFAULT_PROVIDER)


# --- Khởi tạo Flask App ---
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Khởi tạo g4f client (nếu có)
client = Client() if G4F_AVAILABLE else None

# --- Các hàm tiện ích (không thay đổi nhiều) ---

def get_storage_headers():
    headers = {"Content-Type": "application/json"}
    if STORAGE_API_KEY:
        headers["X-Master-Key"] = STORAGE_API_KEY
    return headers

def load_history():
    if not STORAGE_API_URL: return []
    try:
        res = requests.get(f"{STORAGE_API_URL}/latest", headers=get_storage_headers(), timeout=5)
        if res.status_code == 200:
            data = res.json()
            # Hỗ trợ cả định dạng record của JSONBin và mảng thuần
            messages = data.get("record", data if isinstance(data, list) else [])
            return messages[-MAX_HISTORY_MESSAGES:] if isinstance(messages, list) else []
    except Exception as e:
        app.logger.error(f"Failed to load history: {e}")
    return []

def save_history(messages):
    if not STORAGE_API_URL: return False
    try:
        limited_messages = messages[-MAX_HISTORY_MESSAGES:]
        res = requests.put(STORAGE_API_URL, json=limited_messages, headers=get_storage_headers(), timeout=5)
        return res.status_code == 200
    except Exception as e:
        app.logger.error(f"Failed to save history: {e}")
        return False

def build_prompt(messages, user_input):
    now = datetime.now(VIETNAM_TZ).strftime('%A, %d/%m/%Y, %H:%M:%S')
    system_prompt = {"role": "system", "content": f"You are a helpful AI assistant. Current time in Vietnam: {now}."}
    recent_messages = messages[-(MAX_PAYLOAD_MESSAGES - 1):] if messages else []
    return [system_prompt] + recent_messages + [{"role": "user", "content": user_input}]


# --- API Endpoints ---

@app.route("/")
def home():
    return render_template("sv.html")

@app.route("/api/chat", methods=["POST"])
def api_chat_stream():
    """
    Endpoint chính, đã được gia cố để chống crash và báo lỗi rõ ràng.
    """
    try:
        data = request.get_json(silent=True)
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Message is empty"}), 400
        if not client:
            return jsonify({"error": "g4f library not available on server."}), 500

        app.logger.info(f"Received message: {user_message}")

        def generate_response():
            messages = []
            full_response = ""
            try:
                # Tải lịch sử bên trong generator để đảm bảo luôn mới nhất
                messages = load_history()
                payload = build_prompt(messages, user_message)
                
                app.logger.info(f"Using provider: {G4F_PROVIDER.__name__}")
                
                response_stream = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=payload,
                    stream=True,
                    timeout=API_CALL_TIMEOUT,
                    provider=G4F_PROVIDER # <--- MỚI: Chỉ định provider rõ ràng
                )

                for chunk in response_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'delta': content})}\n\n"
            
            except Exception as e:
                # --- MỚI: Bắt lỗi từ g4f và báo về cho client ---
                app.logger.error(f"An error occurred during stream generation: {e}", exc_info=True)
                error_message = f"Xin lỗi, đã có lỗi từ nhà cung cấp AI ({G4F_PROVIDER.__name__}): {str(e)}"
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                # Vẫn lưu lại lỗi để biết ngữ cảnh
                full_response = error_message
            finally:
                # --- MỚI: Luôn luôn lưu lại lịch sử, kể cả khi có lỗi ---
                if user_message and full_response:
                    messages.append({"role": "user", "content": user_message})
                    messages.append({"role": "assistant", "content": full_response.strip()})
                    save_history(messages)
                    app.logger.info("History saved.")

        return Response(generate_response(), mimetype='text/event-stream')

    except Exception as e:
        # --- MỚI: Bắt lỗi toàn cục để server không bao giờ crash ---
        app.logger.error(f"A critical error occurred in /api/chat: {e}", exc_info=True)
        return jsonify({"error": "A critical server error occurred."}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({"messages": load_history()})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    if save_history([]):
        return jsonify({"message": "History cleared"})
    return jsonify({"error": "Failed to clear history"}), 500
