import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thêm thư mục src vào path để import được geolocation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.geolocation import calculate_tank_position_local_frame
except ImportError:
    print("Lỗi: Không tìm thấy module src.geolocation. Hãy đảm bảo bạn đang chạy từ thư mục gốc dự án.")
    sys.exit(1)

# CẬP NHẬT: Đường dẫn file cho ảnh test số 2
MODEL_PATH = "/mnt/d/detect_car/runs/detect/yolov8n_aerial_v3/weights/best.pt"
IMAGE_PATH = "/mnt/d/fpga_drone_dis/photo_test_2.jpg"           # <--- Đã sửa
OUTPUT_IMAGE_PATH = "/mnt/d/fpga_drone_dis/result_photo_test_2.jpg" # <--- Đã sửa

def run_integration_test():
    print("=== BẮT ĐẦU INTEGRATION TEST (YOLO + GEOLOCATION) ===\n")

    # 1. KIỂM TRA FILE
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy model tại {MODEL_PATH}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"Lỗi: Không tìm thấy ảnh tại {IMAGE_PATH}")
        return

    # 2. LOAD MODEL & ẢNH
    print(f"-> Đang load model YOLOv8 từ: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    print(f"-> Đang đọc ảnh: {IMAGE_PATH}...")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Lỗi: Không đọc được ảnh.")
        return
    
    img_h, img_w = img.shape[:2]
    print(f"-> Kích thước ảnh input: {img_w}x{img_h}")

    # 3. NHẬN DIỆN (INFERENCE)
    # Vẫn giữ imgsz=1280 để đảm bảo độ nét tốt nhất, dù ảnh này xe khá to
    print("-> Đang thực hiện nhận diện (imgsz=1280)...")
    
    results = model.predict(
        source=img, 
        imgsz=1280,      
        conf=0.25,       # Tăng conf lên chút vì xe này rõ nét hơn
        iou=0.45,        
        save=False,      
        verbose=True
    )
    
    result = results[0]
    
    if len(result.boxes) == 0:
        print("CẢNH BÁO: Không phát hiện được xe nào!")
        return

    # Lấy box có độ tin cậy cao nhất
    best_box = max(result.boxes, key=lambda x: x.conf[0])
    
    # Trích xuất thông tin
    x_c, y_c, w, h = best_box.xywh[0].cpu().numpy()
    conf = best_box.conf[0].cpu().numpy()
    cls_id = int(best_box.cls[0].cpu().numpy())
    class_name = model.names[cls_id]

    print(f"\n[KẾT QUẢ AI]")
    print(f"  + Tìm thấy: {class_name} ({conf:.2f})")
    print(f"  + Tọa độ tâm (pixel): ({x_c:.1f}, {y_c:.1f})")
    print(f"  + Kích thước box: ({w:.1f}, {h:.1f})")
    
    # Vẽ và lưu ảnh
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.circle(img, (int(x_c), int(y_c)), 8, (0, 0, 255), -1)
    cv2.putText(img, f"{class_name}: {conf:.2f}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(OUTPUT_IMAGE_PATH, img)
    print(f"  + Đã lưu ảnh kết quả tại: {OUTPUT_IMAGE_PATH}")

    # 4. TÍNH TOÁN GEOLOCATION (CẬP NHẬT)
    print(f"\n[TÍNH TOÁN VỊ TRÍ]")
    
    # Cấu hình Drone
    d_x, d_y, d_z = 0.0, 0.0, 0.6  # Drone cao 0.6m
    roll, pitch, yaw = 0.0, -45.0, 0.0 # Camera góc -45 độ
    
    # --- LOGIC SCALE TIÊU CỰ MỚI ---
    # Thông số gốc (Reference) từ ảnh 2560x1440
    ref_w = 2560.0
    ref_fx = 2474.0
    
    # Tính tỷ lệ scale dựa trên chiều rộng (Width)
    # Vì ảnh mới là 2048px, nhỏ hơn 2560px
    scale_factor = img_w / ref_w
    
    # Tính fx mới
    fx_new = ref_fx * scale_factor
    
    # Với ảnh tỷ lệ 4:3 (2048x1536), ta giả định pixel hình vuông (Square Pixels)
    # nên fx và fy sẽ xấp xỉ bằng nhau.
    fy_new = fx_new 
    
    print(f"  + Tiêu cự tính toán (fx, fy): {fx_new:.1f}, {fy_new:.1f}")
    
    # Gọi hàm tính toán
    pos, msg = calculate_tank_position_local_frame(
        u=x_c, v=y_c,
        d_x=d_x, d_y=d_y, d_z=d_z,
        roll=roll, pitch=pitch, yaw=yaw,
        img_w=img_w, img_h=img_h,
        fx=fx_new, fy=fy_new
    )

    if pos:
        real_x, real_y, real_z = pos
        print(f"  -> VỊ TRÍ TÍNH ĐƯỢC: X={real_x:.3f}m, Y={real_y:.3f}m, Z={real_z:.3f}m")
        print(f"  -> VỊ TRÍ KỲ VỌNG : X=0.600m, Y=0.000m, Z=0.000m")
        
        # Đánh giá sai số
        error = np.sqrt((real_x - 0.6)**2 + (real_y - 0)**2)
        print(f"  -> Sai số: {error:.3f}m")
    else:
        print(f"  -> Lỗi tính toán: {msg}")

if __name__ == "__main__":
    run_integration_test()