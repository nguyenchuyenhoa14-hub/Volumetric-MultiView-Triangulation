import sys
import os
import numpy as np
import math
import plotly.graph_objects as go

# Nhập hàm từ file geolocation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.geolocation import triangulate_frustum_centroid, get_sight_vector, get_bbox_keypoints

# --- CONFIG CAMERA ---
IMG_W, IMG_H = 2048, 1536
FX, FY = 1979.2, 1979.2

# Cài đặt Vị trí Camera
pos1 = np.array([0.0, 0.0, 0.6])
att1 = (0.0, -45.0, 0.0)

pos2 = np.array([0.6, 1.0, 1.0])
att2 = (0.0, -45.0, -90.0)

# Tâm xe thật sự (True Target Center)
car_center = np.array([0.6, 0.0, 0.0])

# Hàm phụ trợ để tính pixel cho các điểm bị lệch (Perspective Shift)
def get_rotation_matrix(roll, pitch, yaw):
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return R_z @ R_y @ R_x

def world_to_pixel(target_pos, cam_pos, roll, pitch, yaw, img_w, img_h, fx, fy):
    v_world = np.array(target_pos) - np.array(cam_pos)
    R_inv = np.linalg.inv(get_rotation_matrix(roll, pitch, yaw))
    v_body = R_inv @ v_world
    v_body_scaled = v_body / v_body[0]
    u = v_body_scaled[1] * fx + (img_w / 2)
    v = -v_body_scaled[2] * fy + (img_h / 2)
    return (u, v)

def run_frustum_centroid_simulation():
    print("=== 3D SIMULATION: MULTI-RAY FRUSTUM CENTROID ===\n")

    # --- 1. MÔ PHỎNG SỰ LỆCH TÂM (PERSPECTIVE SHIFT) ---
    # Ảnh 1: YOLO bắt vào cản sau
    physical_pt1 = np.array([0.5, 0.0, 0.0]) 
    uv1_center = world_to_pixel(physical_pt1, pos1, *att1, IMG_W, IMG_H, FX, FY)
    
    # Ảnh 2: YOLO bắt vào cửa xe
    physical_pt2 = np.array([0.6, 0.1, 0.0]) 
    uv2_center = world_to_pixel(physical_pt2, pos2, *att2, IMG_W, IMG_H, FX, FY)

    # Khởi tạo Bounding Box quanh các tâm bị lệch này
    bbox1 = (uv1_center[0], uv1_center[1], 500, 300) 
    bbox2 = (uv2_center[0], uv2_center[1], 700, 250)

    # --- 2. TÍNH TOÁN PHƯƠNG PHÁP MỚI (Trọng tâm Frustum 5 điểm) ---
    new_centroid, msg, point_cloud = triangulate_frustum_centroid(
        pos1, att1, bbox1,
        pos2, att2, bbox2,
        IMG_W, IMG_H, FX, FY
    )
    new_error = np.linalg.norm(new_centroid - car_center)

    print("[CALCULATION RESULTS]")
    print(f"  True Car Center    : [{car_center[0]:.3f}, {car_center[1]:.3f}, {car_center[2]:.3f}]")
    print(f"  Estimated Centroid : [{new_centroid[0]:.3f}, {new_centroid[1]:.3f}, {new_centroid[2]:.3f}]")
    print("\n[ACCURACY]")
    print(f"  Position Error     : {new_error:.4f} meters")

    # --- 3. VẼ BIỂU ĐỒ TRỰC QUAN ---
    fig = go.Figure()

    # Ground
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 10), np.linspace(-0.2, 1.2, 10))
    fig.add_trace(go.Surface(x=xx, y=yy, z=np.zeros_like(xx), opacity=0.1, colorscale='Greens', showscale=False))

    # Helper function để vẽ Frustum
    def draw_frustum(pos, att, bbox, color, name, ray_len=1.5):
        pts = get_bbox_keypoints(*bbox)
        vertices = [pos]
        for uv in pts[1:]: # Lấy 4 góc
            ray = get_sight_vector(*uv, *att, IMG_W, IMG_H, FX, FY)
            vertices.append(pos + ray * ray_len)
        vertices = np.array(vertices)
        
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = [0, 0, 0, 0, 1, 1], [1, 2, 3, 4, 2, 3], [2, 3, 4, 1, 3, 4]
        
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=0.1, name=name))
        
        edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]], y=[y[edge[0]], y[edge[1]]], z=[z[edge[0]], z[edge[1]]],
                mode='lines', line=dict(color=color, width=1), showlegend=False
            ))

    draw_frustum(pos1, att1, bbox1, 'blue', 'Photo 1 Frustum')
    draw_frustum(pos2, att2, bbox2, 'red', 'Photo 2 Frustum')

    # Vẽ Camera
    fig.add_trace(go.Scatter3d(x=[pos1[0]], y=[pos1[1]], z=[pos1[2]], mode='markers+text', marker=dict(size=8, color='blue', symbol='square'), name='Photo 1', text=['Photo 1'], textposition='top center'))
    fig.add_trace(go.Scatter3d(x=[pos2[0]], y=[pos2[1]], z=[pos2[2]], mode='markers+text', marker=dict(size=8, color='red', symbol='square'), name='Photo 2', text=['Photo 2'], textposition='top center'))

    # Vẽ Đám mây điểm (5 điểm giao cắt)
    point_cloud_arr = np.array(point_cloud)
    fig.add_trace(go.Scatter3d(
        x=point_cloud_arr[:, 0], y=point_cloud_arr[:, 1], z=point_cloud_arr[:, 2],
        mode='markers', marker=dict(size=4, color='orange', symbol='circle'),
        name='5 Intersected Points'
    ))

    # Vẽ Tâm thực tế
    fig.add_trace(go.Scatter3d(
        x=[car_center[0]], y=[car_center[1]], z=[car_center[2]],
        mode='markers+text', marker=dict(size=10, color='black', symbol='diamond'),
        name='True Car Center', text=['True Center'], textposition='top right',
        textfont=dict(color='black')
    ))

    # Vẽ Điểm ước lượng Trọng tâm Mới
    fig.add_trace(go.Scatter3d(
        x=[new_centroid[0]], y=[new_centroid[1]], z=[new_centroid[2]],
        mode='markers+text', marker=dict(size=8, color='green', symbol='circle'),
        name='Estimated Centroid', text=['Est. Centroid'], textposition='bottom left',
        textfont=dict(color='green')
    ))

    # Formatting đồ thị
    fig.update_layout(
        title=dict(
            text="<b>Multi-Ray Frustum Centroid Estimation</b><br>Correcting Perspective Shift",
            font=dict(size=28)
        ),
        legend=dict(font=dict(size=24)),
        scene=dict(
            xaxis_title='X Axis (meters)', yaxis_title='Y Axis (meters)', zaxis_title='Z Axis (meters)',
            xaxis=dict(range=[-0.2, 1.2]), yaxis=dict(range=[-0.2, 1.2]), zaxis=dict(range=[-0.4, 1.2]),
            aspectmode='cube'
        ),
        annotations=[
            dict(text=f"<b>Position Error: {new_error:.4f} m</b>", showarrow=False, xref="paper", yref="paper", x=0.02, y=0.95, font=dict(color="green", size=20))
        ],
        margin=dict(l=0, r=0, b=0, t=60)
    )

    fig.show()

if __name__ == "__main__":
    run_frustum_centroid_simulation()
