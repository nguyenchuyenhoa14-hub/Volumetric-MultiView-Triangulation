import sys
import os
import numpy as np
import math

# Thêm đường dẫn src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.geolocation import triangulate_3d_point

# --- CẤU HÌNH THỰC TẾ TỪ ẢNH 2 (Dựa trên kết quả YOLO Integration) ---
IMG_W, IMG_H = 2048, 1536
FX, FY = 1979.2, 1979.2

def run_stereo_test_real_data():
    print("=== TEST TÍNH TỌA ĐỘ Z BẰNG 2 ẢNH (REAL DATA CALIBRATION) ===\n")
    
    # 1. THÔNG SỐ VIEW 1 (Lấy từ kết quả thật của photo_test_2.jpg)
    pos1 = (0.0, 0.0, 0.6)
    att1 = (0.0, -45.0, 0.0) 
    uv1 = (1062.3, 766.9) # Tọa độ YOLO nhận diện được
    
    # Target thực tế tính được từ View 1 (Z = 0)
    # Lấy đúng kết quả từ log của bạn
    target_true = np.array([0.601, 0.016, 0.000])
    print(f"-> Vị trí vật thể mục tiêu (Ground Truth): {target_true}")
    print(f"[VIEW 1] Pos={pos1}, Att={att1}, Pixel={uv1}\n")

    # 2. XÂY DỰNG VIEW 2 (Drone dịch chuyển sang phải 0.3m)
    pos2 = (0.0, 0.3, 0.6)
    att2 = (0.0, -45.0, 0.0)
    
    # TÍNH TOÁN NGƯỢC PIXEL (U2, V2) CHUẨN XÁC THEO HÌNH HỌC
    # Vector từ Drone 2 đến Target
    v_world = target_true - np.array(pos2)
    
    # Ma trận quay R_y góc +45 độ (Ngược lại của Pitch -45 để chuyển từ World về Body)
    p = math.radians(45.0)
    R_y_inv = np.array([
        [np.cos(p), 0, -np.sin(p)],
        [0,         1, 0],
        [np.sin(p), 0, np.cos(p)]
    ])
    
    # Vector trong hệ trục Body của Drone 2
    v_body = R_y_inv @ v_world
    
    # Chuẩn hóa về dạng [1, xc, -yc]
    v_body_scaled = v_body / v_body[0]
    x_c = v_body_scaled[1]
    y_c = -v_body_scaled[2]
    
    # Quy đổi về Pixel
    cx, cy = IMG_W / 2, IMG_H / 2
    u2 = x_c * FX + cx
    v2 = y_c * FY + cy
    uv2 = (u2, v2)
    
    print(f"[VIEW 2] Pos={pos2}, Att={att2}")
    print(f"         Pixel tính toán chuẩn xác={uv2}\n")

    # 3. CHẠY THUẬT TOÁN TRIANGULATION
    print("-> Đang tính giao điểm 2 tia bằng thuật toán...")
    estimated_point, msg = triangulate_3d_point(
        pos1, att1, uv1,
        pos2, att2, uv2,
        img_w=IMG_W, img_h=IMG_H,
        fx=FX, fy=FY
    )
    
    if estimated_point is not None:
        X, Y, Z = estimated_point
        print(f"\n[KẾT QUẢ CUỐI CÙNG]")
        print(f"  Vị trí Triangulation: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
        
        err = np.linalg.norm(estimated_point - target_true)
        print(f"  Sai số so với View 1 : {err:.4f} m")
        
        if abs(Z) < 0.01:
            print("\n=> THÀNH CÔNG RỰC RỠ! Thuật toán đã tự động tính ra độ cao Z=0 mà không cần truyền biến giả định.")
        else:
            print(f"\n=> Thất bại, Z lệch: {Z:.3f}m")
    else:
        print(f"Lỗi: {msg}")

if __name__ == "__main__":
    run_stereo_test_real_data()