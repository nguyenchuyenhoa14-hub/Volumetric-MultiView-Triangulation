import sys
import os
import numpy as np

# Thêm đường dẫn src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.geolocation import triangulate_3d_point

# --- CẤU HÌNH CAMERA (Lấy theo tỷ lệ ảnh thực tế của bạn) ---
IMG_W, IMG_H = 2048, 1536
FX, FY = 1979.2, 1979.2

def run_stereo_perfect_math():
    print("=== TEST TOÁN HỌC STEREO VISION TỐI ƯU ===\n")
    
    target_true = np.array([0.6, 0.0, 0.0])
    print(f"-> Vị trí vật thể mục tiêu (Ground Truth): {target_true}\n")

    # --- 1. VIEW 1 ---
    # Drone ở gốc tọa độ, cao 0.6m. Nhìn thẳng (Yaw=0), chúi xuống (Pitch=-45).
    pos1 = (0.0, 0.0, 0.6)
    att1 = (0.0, -45.0, 0.0) 
    uv1 = (IMG_W / 2, IMG_H / 2) # Xe ở ngay tâm ảnh
    
    print(f"[VIEW 1]")
    print(f"  Pos: {pos1} | Att (R,P,Y): {att1} | Pixel: {uv1}")

    # --- 2. VIEW 2 ---
    # Drone bay sang ngang và tiến lên (0.6, 0.6, 0.6).
    # Để nhìn thấy xe ở (0.6, 0, 0), drone quay mũi sang TRẢI/PHẢI 90 độ (Yaw = -90).
    # Lúc này ống kính chĩa thẳng vào trục Y âm. Xe CŨNG nằm ngay tâm ảnh.
    pos2 = (0.6, 0.6, 0.6)
    att2 = (0.0, -45.0, -90.0) 
    uv2 = (IMG_W / 2, IMG_H / 2) 
    
    print(f"\n[VIEW 2]")
    print(f"  Pos: {pos2} | Att (R,P,Y): {att2} | Pixel: {uv2}\n")

    # --- 3. TÍNH TOÁN GIAO ĐIỂM ---
    print("-> Đang tính giao điểm 2 tia (Triangulation)...")
    estimated_point, msg = triangulate_3d_point(
        pos1, att1, uv1,
        pos2, att2, uv2,
        img_w=IMG_W, img_h=IMG_H, fx=FX, fy=FY
    )

    if estimated_point is not None:
        X, Y, Z = estimated_point
        print(f"\n[KẾT QUẢ CUỐI CÙNG]")
        print(f"  Vị trí Triangulation : X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m")
        
        err = np.linalg.norm(estimated_point - target_true)
        print(f"  Sai số tổng thể      : {err:.4f} m")
        
        if err < 0.001:
            print("\n=> XUẤT SẮC! Thuật toán hội tụ hoàn hảo. 2 vector cắt nhau chính xác 100%.")
    else:
        print(f"Lỗi: {msg}")

if __name__ == "__main__":
    run_stereo_perfect_math()