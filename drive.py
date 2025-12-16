import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
import base64
from io import BytesIO
from PIL import Image
import cv2 
import math

sio = socketio.Server() # Tạo server SocketIO
app = Flask(__name__) # Tạo ứng dụng Flask
# Giới hạn tốc độ tối đa 
speed_limit = 15 #10 

# --- CẤU HÌNH ---
# Kích thước vô lăng trên màn hình
WHEEL_SIZE = (120, 120) 
# Hệ số phóng to màn hình dashboard
DASHBOARD_SCALE = 1.2 

# --- TẢI ẢNH VÔ LĂNG ---
wheel_img = cv2.imread('wheel.png', cv2.IMREAD_UNCHANGED) 
if wheel_img is None: 
    print("CẢNH BÁO: Không tìm thấy file 'wheel.png'.") 
else:
    wheel_img = cv2.resize(wheel_img, WHEEL_SIZE) 

# --- CÁC HÀM PHỤ TRỢ ---

def rotate_and_overlay(background, foreground, angle, x_offset, y_offset):
    h, w = foreground.shape[:2] 
    center = (w // 2, h // 2)
    # Nhân với -180 (hoặc -360) tùy vào độ nhạy 
    rotation_matrix = cv2.getRotationMatrix2D(center, angle * -180, 1.0) 
    rotated_wheel = cv2.warpAffine(foreground, rotation_matrix, (w, h))
    
    wheel_bgr = rotated_wheel[:, :, :3]
    wheel_alpha = rotated_wheel[:, :, 3] / 255.0

    h_bg, w_bg = background.shape[:2]
    roi = background[y_offset:y_offset+h, x_offset:x_offset+w] 

    for c in range(0, 3):
        roi[:, :, c] = (wheel_alpha * wheel_bgr[:, :, c] + 
                        (1.0 - wheel_alpha) * roi[:, :, c])

    background[y_offset:y_offset+h, x_offset:x_offset+w] = roi
    return background

def create_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# --- HÀM VẼ DASHBOARD ---
def draw_dashboard(original_image, steering_angle, speed, throttle):
    # 1. Chuyển hệ màu từ RGB (PIL) sang BGR (OpenCV)
    display_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # 2. Vẽ vô lăng
    if wheel_img is not None:
        h_img, w_img = display_img.shape[:2]
        h_wheel, w_wheel = wheel_img.shape[:2]
        
        # Đặt vô lăng ở góc dưới bên phải, cách lề 20px
        x_pos = w_img - w_wheel - 20
        y_pos = h_img - h_wheel - 20
        
        display_img = rotate_and_overlay(display_img, wheel_img, steering_angle, x_pos, y_pos)

    # 3. Vẽ thông số
    # Vẽ nền đen mờ cho chữ dễ đọc (tùy chọn)
    # cv2.rectangle(display_img, (0, 0), (250, 80), (0, 0, 0), -1) 
    
    cv2.putText(display_img, f"Angle: {steering_angle:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_img, f"Speed: {speed:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Vẽ thanh Ga (Throttle Bar) cho trực quan
    # bar_width = int(throttle * 100)
    # cv2.rectangle(display_img, (10, 80), (10 + bar_width, 95), (0, 0, 255), -1)
    # cv2.rectangle(display_img, (10, 80), (110, 95), (255, 255, 255), 1) # Khung viền

    # 4. Phóng to ảnh
    display_img = cv2.resize(display_img, None, fx=DASHBOARD_SCALE, fy=DASHBOARD_SCALE, interpolation=cv2.INTER_CUBIC)
    
    return display_img

# --- SERVER EVENT HANDLERS ---

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            speed = float(data['speed'])
            image_str = data['image']
            image = Image.open(BytesIO(base64.b64decode(image_str)))
            original_image_np = np.asarray(image) 
            
            # --- PHẦN ĐIỀU KHIỂN ---
            processed_image = img_preprocess(original_image_np)
            image_input = np.array([processed_image])
            steering_angle = float(model.predict(image_input, verbose=0))
            
            # --- ĐIỀU CHỈNH GA ---
            throttle = 1.0 - speed/speed_limit
            # throttle = 1.2 - (steering_angle**2) - (speed / speed_limit) # Giúp xe ổn định hơn khi vào cua
            
            print(f'Góc lái: {steering_angle:.4f} | Ga: {throttle:.2f}')
            send_control(steering_angle, throttle)

            # --- PHẦN HIỂN THỊ ---
            dashboard_img = draw_dashboard(original_image_np, steering_angle, speed, throttle)
            cv2.imshow("Camera hanh trinh", dashboard_img)
            cv2.waitKey(1)

        except Exception as e:
            print(f"Lỗi: {e}")
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    print("Đang khởi tạo kiến trúc model...")
    model = create_model()
    
    # ĐẶT TÊN FILE MODEL Ở ĐÂY 
    weights_file = 'model/best_model_weights_2_tracks_no3.npy'
    
    print(f"Đang nạp trọng số từ: {weights_file} ...")
    try:
        weights_data = np.load(weights_file, allow_pickle=True)
        model.set_weights(weights_data)
        print("-> Đã nạp trọng số thành công!")
    except Exception as e:
        print(f"!!! LỖI NẠP TRỌNG SỐ: {e}")
        exit()

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)