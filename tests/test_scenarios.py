import sys
import os
# Thêm thư mục cha vào đường dẫn để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geolocation import calculate_tank_position_local_frame

def run_test_cases():
    print("=== BẮT ĐẦU CHẠY TEST CASE GEOLOCATION ===\n")

    # --- CASE 1: CƠ BẢN (Chuẩn) ---
    # Drone cao 100m, nhìn xuống 45 độ. Xe tăng ở giữa ảnh.
    # Kỳ vọng: Xe tăng cách 100m về phía trước (X=100, Y=0).
    print("Test Case 1: Cơ bản (Pitch -45, H=100m)")
    pos, msg = calculate_tank_position_local_frame(
        u=1280, v=720,          # Tâm ảnh
        d_x=0, d_y=0, d_z=100,  # Vị trí drone
        roll=0, pitch=-45, yaw=0
    )
    print(f"  -> Kết quả: {pos}")
    print(f"  -> Kỳ vọng: (100.0, 0.0, 0.0) [Xấp xỉ]")
    print("-" * 30)

    # --- CASE 2: NHÌN THẲNG ĐỨNG (Top-down) ---
    # Drone cao 50m, nhìn thẳng xuống (-90 độ).
    # Kỳ vọng: Xe tăng trùng tọa độ XY với Drone (X=0, Y=0).
    print("Test Case 2: Nhìn thẳng đứng (Pitch -90, H=50m)")
    pos, msg = calculate_tank_position_local_frame(
        u=1280, v=720,
        d_x=10, d_y=20, d_z=50, # Drone đang ở (10, 20)
        roll=0, pitch=-90, yaw=0
    )
    print(f"  -> Kết quả: {pos}")
    print(f"  -> Kỳ vọng: (10.0, 20.0, 0.0)")
    print("-" * 30)

    # --- CASE 3: XOAY NGANG (Yaw 90 độ - Hướng Đông) ---
    # Drone quay mặt sang phải 90 độ, chúi 45 độ.
    # Kỳ vọng: Xe tăng nằm trên trục Y (Phía Đông), X xấp xỉ 0.
    print("Test Case 3: Quay sang Đông (Yaw 90, Pitch -45)")
    pos, msg = calculate_tank_position_local_frame(
        u=1280, v=720,
        d_x=0, d_y=0, d_z=100,
        roll=0, pitch=-45, yaw=90
    )
    print(f"  -> Kết quả: {pos}")
    print(f"  -> Kỳ vọng: (0.0, 100.0, 0.0) [Xấp xỉ]")
    print("-" * 30)

    # --- CASE 4: LỆCH TÂM ẢNH & ĐỊA HÌNH ĐỒI ---
    # Xe tăng nằm lệch phải trong ảnh (u > 1280).
    # Địa hình mục tiêu là đồi cao 20m.
    print("Test Case 4: Lệch tâm + Địa hình cao (Z_target=20m)")
    pos, msg = calculate_tank_position_local_frame(
        u=1280 + 500, v=720,    # Lệch phải 500px
        d_x=0, d_y=0, d_z=100,
        roll=0, pitch=-45, yaw=0,
        z_target_area=20.0      # Đồi cao 20m
    )
    # Phân tích: 
    # Drone cao thực tế so với đồi là 80m.
    # Góc pitch 45 độ -> Khoảng cách phía trước ~ 80m.
    # Lệch phải trong ảnh -> Tọa độ Y sẽ dương.
    print(f"  -> Kết quả: {pos}")
    print("-" * 30)

    # --- CASE 5: LỖI (Nhìn lên trời) ---
    print("Test Case 5: Nhìn lên trời (Pitch +10)")
    pos, msg = calculate_tank_position_local_frame(
        u=1280, v=720,
        d_x=0, d_y=0, d_z=100,
        roll=0, pitch=10, yaw=0 
    )
    print(f"  -> Kết quả: {pos}")
    print(f"  -> Thông báo lỗi: {msg}")
    print("=" * 30)

if __name__ == "__main__":
    run_test_cases()