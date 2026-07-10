# File: tests/test_tiling_yolo.py
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Kiểm tra torchvision (cần cho NMS)
try:
    import torchvision
except ImportError:
    print("Lỗi: Thiếu thư viện torchvision. Hãy cài đặt bằng lệnh: pip install torchvision")
    sys.exit(1)

# --- CẤU HÌNH (Đường dẫn tuyệt đối - Không cần sửa) ---
MODEL_PATH = "/mnt/d/detect_car/runs/detect/yolov8n_aerial_v3/weights/best.pt"
IMAGE_PATH = "/mnt/d/fpga_drone_dis/photo_test_10.jpg"
OUTPUT_PATH = "/mnt/d/fpga_drone_dis/result_photo_test_10.jpg"

# --- CẤU HÌNH THUẬT TOÁN ---
TILE_SIZE = 1280   # Kích thước cắt (nên lớn hơn 416 để bao quát tốt hơn)
INPUT_SIZE = 416  # Kích thước input thực tế vào model (giả lập FPGA)
OVERLAP = 0.2     # 20% chồng lấn
CONF_THRES = 0.15 # Ngưỡng chấp nhận object
IOU_THRES = 0.3   # Ngưỡng gộp box (thấp để gộp các box trùng ở vùng cắt)
ANNOTATE_TOP1_ONLY = True  # True: chỉ vẽ 1 box có score cao nhất để làm ảnh minh họa

# --- CẤU HÌNH HIỂN THỊ 5 ĐIỂM TRÊN BOUNDING BOX ---
POINT_RADIUS = 8
POINT_COLORS = {
    "CENTER": (0, 0, 255),   # Đỏ
    "TL": (255, 255, 0),     # Vàng nhạt
    "TR": (255, 0, 255),     # Tím hồng
    "BL": (0, 255, 255),     # Vàng
    "BR": (255, 128, 0),     # Cam
}

def draw_five_keypoints(img, x1, y1, x2, y2):
    """Vẽ 5 điểm đặc trưng của bounding box: tâm + 4 góc."""
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    points = {
        "CENTER": (cx, cy),
        "TL": (x1, y1),
        "TR": (x2, y1),
        "BL": (x1, y2),
        "BR": (x2, y2),
    }

    for label, (px, py) in points.items():
        color = POINT_COLORS[label]
        cv2.circle(img, (px, py), POINT_RADIUS, color, -1)
        cv2.circle(img, (px, py), POINT_RADIUS + 2, (255, 255, 255), 2)
        cv2.putText(
            img,
            label,
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return points

def get_slices(img_h, img_w, tile_size, overlap):
    """Tính toán tọa độ các mảnh cắt (Sliding Window)"""
    step = int(tile_size * (1 - overlap))
    slices = []
    
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            # Tính chiều rộng/cao thực tế của mảnh (xử lý phần rìa ảnh)
            h_real = min(tile_size, img_h - y)
            w_real = min(tile_size, img_w - x)
            
            # Logic xử lý mép ảnh:
            # Nếu mảnh ở mép nhỏ hơn tile_size, ta lùi x, y lại để lấy đủ tile_size
            # (Tránh việc resize các mảnh nhỏ xíu ở rìa làm méo hình quá mức)
            x_start = x
            y_start = y
            
            if h_real < tile_size and y_start > 0:
                y_start = max(0, img_h - tile_size)
            if w_real < tile_size and x_start > 0:
                x_start = max(0, img_w - tile_size)
            
            slices.append((x_start, y_start, tile_size, tile_size))
            
            # Nếu đã đến mép ảnh thì dừng loop x
            if x + w_real >= img_w:
                break
            x += step
        
        # Nếu đã đến mép ảnh thì dừng loop y
        if y + h_real >= img_h:
            break
        y += step
        
    return slices

def run_tiled_inference():
    print("=== BẮT ĐẦU TEST TILING (CẮT ẢNH) ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy model tại {MODEL_PATH}")
        return
    
    print(f"-> Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f"-> Đọc ảnh: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Lỗi: Không đọc được ảnh.")
        return
        
    H, W = img.shape[:2]
    print(f"-> Kích thước gốc: {W}x{H}")

    # 1. TẠO CÁC MẢNH CẮT (TILES)
    slices = get_slices(H, W, tile_size=TILE_SIZE, overlap=OVERLAP)
    print(f"-> Đã chia thành {len(slices)} mảnh (Tile size: {TILE_SIZE}x{TILE_SIZE})")

    all_boxes = []
    all_scores = []
    all_classes = []

    # 2. CHẠY MODEL TRÊN TỪNG MẢNH
    print("-> Đang chạy inference trên từng mảnh...")
    for i, (x, y, w, h) in enumerate(slices):
        # Cắt ảnh con
        # Lưu ý: cần đảm bảo không cắt quá giới hạn mảng
        tile_img = img[y:y+h, x:x+w]
        
        # Nếu ảnh cắt bị rỗng (do lỗi logic nào đó), bỏ qua
        if tile_img.size == 0: continue

        # Inference
        # imgsz=INPUT_SIZE (416) -> Giả lập việc FPGA chỉ nhận ảnh 416x416
        results = model.predict(tile_img, imgsz=INPUT_SIZE, conf=CONF_THRES, verbose=False)
        
        # 3. MAPPING (CHUYỂN TỌA ĐỘ VỀ HỆ TRỤC ẢNH GỐC 2K)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Tọa độ xyxy trên mảnh nhỏ
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Cộng thêm tọa độ gốc (offset) của mảnh cắt để ra tọa độ trên ảnh lớn
                real_x1 = bx1 + x
                real_y1 = by1 + y
                real_x2 = bx2 + x
                real_y2 = by2 + y
                
                all_boxes.append([real_x1, real_y1, real_x2, real_y2])
                all_scores.append(conf)
                all_classes.append(cls)

    if not all_boxes:
        print("CẢNH BÁO: Không tìm thấy xe nào sau khi cắt ảnh!")
        return

    # 4. NMS (GỘP CÁC BOX TRÙNG LẶP TOÀN CỤC)
    print(f"-> Tìm thấy tổng cộng {len(all_boxes)} box sơ bộ. Đang gộp (NMS)...")
    
    boxes_tensor = torch.tensor(all_boxes)
    scores_tensor = torch.tensor(all_scores)
    
    # Dùng NMS của torchvision (chuẩn và nhanh)
    # iou_threshold thấp (0.3) để gộp mạnh tay các box trùng nhau ở vùng giao thoa
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=IOU_THRES)
    
    print(f"-> Kết quả: Giữ lại {len(keep_indices)} box duy nhất.")

    # 5. VẼ KẾT QUẢ CUỐI CÙNG
    indices_to_draw = [idx.item() for idx in keep_indices]
    if ANNOTATE_TOP1_ONLY and indices_to_draw:
        indices_to_draw = [max(indices_to_draw, key=lambda i: all_scores[i])]
        print("-> Chế độ minh họa: chỉ vẽ 1 box tốt nhất (top-1 confidence).")

    for idx in indices_to_draw:
        x1, y1, x2, y2 = map(int, all_boxes[idx])
        score = all_scores[idx]
        cls_id = int(all_classes[idx])
        name = model.names[cls_id]
        
        # Tính tâm
        cx, cy = (x1+x2)/2, (y1+y2)/2
        print(f"  + FINAL OBJECT: {name} ({score:.2f}) tại Center=({cx:.1f}, {cy:.1f})")

        # Vẽ khung
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        keypoints = draw_five_keypoints(img, x1, y1, x2, y2)
        cv2.putText(img, f"{name}: {score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(
            "    - 5 điểm: "
            f"CENTER={keypoints['CENTER']}, "
            f"TL={keypoints['TL']}, TR={keypoints['TR']}, "
            f"BL={keypoints['BL']}, BR={keypoints['BR']}"
        )

    cv2.imwrite(OUTPUT_PATH, img)
    print(f"-> Đã lưu ảnh kết quả tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_tiled_inference()
