import numpy as np
import math

def get_sight_vector(u, v, roll, pitch, yaw, img_w, img_h, fx, fy):
    """
    Hàm phụ trợ: Tính vector chỉ phương (v_world) từ pixel (u,v) và tư thế Drone.
    """
    cx, cy = img_w / 2, img_h / 2
    
    # 1. Pixel to Camera Vector
    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    v_body = np.array([1.0, x_c, -y_c]) # Forward(X)=1, Right(Y)=xc, Up(Z)=-yc

    # 2. Rotation Matrix (Body to World)
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    
    R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    
    # Chuyển sang World Frame
    v_world = R_z @ R_y @ R_x @ v_body
    
    # Chuẩn hóa vector
    norm = np.linalg.norm(v_world)
    if norm == 0: return np.array([1, 0, 0])
    return v_world / norm

def triangulate_3d_point(pos1, att1, uv1, pos2, att2, uv2, img_w=2560, img_h=1440, fx=2474.0, fy=2784.0):
    """
    Tính giao điểm 3D của 2 tia nhìn từ 2 vị trí chụp.
    """
    v1 = get_sight_vector(uv1[0], uv1[1], att1[0], att1[1], att1[2], img_w, img_h, fx, fy)
    v2 = get_sight_vector(uv2[0], uv2[1], att2[0], att2[1], att2[2], img_w, img_h, fx, fy)
    
    P1 = np.array(pos1)
    P2 = np.array(pos2)
    
    w0 = P1 - P2
    
    a = np.dot(v1, v1)
    b = np.dot(v1, v2)
    c = np.dot(v2, v2)
    d = np.dot(v1, w0)
    e = np.dot(v2, w0)
    
    denominator = a*c - b*b
    
    if denominator < 1e-6:
        return None, "Hai tia nhìn song song, không thể tính giao điểm"
    
    t = (b*e - c*d) / denominator
    s = (a*e - b*d) / denominator
    
    point_on_1 = P1 + t * v1
    point_on_2 = P2 + s * v2
    
    mid_point = (point_on_1 + point_on_2) / 2.0
    
    if t < 0 or s < 0:
        return mid_point, "CẢNH BÁO: Điểm hội tụ nằm phía sau lưng camera"
        
    return mid_point, "Success"

# =========================================================================
# PHẦN NÂNG CẤP: GIAO CẮT HÌNH CHÓP (FRUSTUM CENTROID ESTIMATION)
# =========================================================================

def get_bbox_keypoints(center_u, center_v, width, height):
    """
    Lấy 5 điểm pixel đại diện cho một Bounding Box: [Tâm, TL, TR, BR, BL]
    """
    w_half, h_half = width / 2.0, height / 2.0
    return [
        (center_u, center_v),                         # Center
        (center_u - w_half, center_v - h_half),       # Top-Left (TL)
        (center_u + w_half, center_v - h_half),       # Top-Right (TR)
        (center_u + w_half, center_v + h_half),       # Bottom-Right (BR)
        (center_u - w_half, center_v + h_half)        # Bottom-Left (BL)
    ]

def triangulate_frustum_centroid(pos1, att1, bbox1, pos2, att2, bbox2, img_w=2560, img_h=1440, fx=2474.0, fy=2784.0):
    """
    Tính tọa độ tối ưu bằng cách lấy trung bình cộng 5 cặp tia giao cắt từ Bounding Box.
    Input: bbox = (center_u, center_v, width, height)
    Output: centroid_3d, message, list_of_intersected_points
    """
    pts1 = get_bbox_keypoints(*bbox1)
    pts2 = get_bbox_keypoints(*bbox2)
    
    valid_3d_points = []
    
    # Nối 5 cặp tia (Tâm-Tâm, Trái Trên - Trái Trên, Phải Dưới - Phải Dưới...)
    for uv1, uv2 in zip(pts1, pts2):
        pt_3d, msg = triangulate_3d_point(
            pos1, att1, uv1, 
            pos2, att2, uv2, 
            img_w, img_h, fx, fy
        )
        
        # Chỉ lấy những điểm hợp lệ (ở phía trước 2 camera)
        if pt_3d is not None and "CẢNH BÁO" not in msg:
            valid_3d_points.append(pt_3d)
            
    if len(valid_3d_points) == 0:
        return None, "Error: No valid intersection points found.", []
        
    # Tính Trọng tâm (Centroid) của đám mây điểm
    valid_3d_points = np.array(valid_3d_points)
    centroid_3d = np.mean(valid_3d_points, axis=0)
    
    return centroid_3d, f"Success (Averaged {len(valid_3d_points)} points)", valid_3d_points

# --- HÀM CŨ ---
def calculate_tank_position_local_frame(u, v, d_x, d_y, d_z, roll, pitch, yaw, z_target_area=0.0, img_w=2560, img_h=1440, fx=2474.0, fy=2784.0):
    v_world = get_sight_vector(u, v, roll, pitch, yaw, img_w, img_h, fx, fy)
    if v_world[2] >= -0.001: return None, "Camera nhìn lên trời"
    k = (z_target_area - d_z) / v_world[2]
    if k < 0: return None, "Mục tiêu phía sau"
    return (d_x + k*v_world[0], d_y + k*v_world[1], z_target_area), "Success"