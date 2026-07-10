import sys
import os
import numpy as np
import math
import plotly.graph_objects as go

# --- 1. IMPORT FUNCTION FROM GEOLOCATION.PY ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.geolocation import triangulate_3d_point

# --- 2. PHYSICS SIMULATION FUNCTIONS ---
def get_rotation_matrix(roll, pitch, yaw):
    """Returns the rotation matrix from Body Frame to World Frame."""
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return R_z @ R_y @ R_x

def world_to_pixel(target_pos, cam_pos, roll, pitch, yaw, img_w, img_h, fx, fy):
    """Simulates camera capturing a 3D point and returns pixel coordinates (u, v)."""
    v_world = np.array(target_pos) - np.array(cam_pos)
    
    R = get_rotation_matrix(roll, pitch, yaw)
    R_inv = np.linalg.inv(R) # Inverse matrix (World to Body)
    
    v_body = R_inv @ v_world
    
    # Normalize along X axis (Camera forward axis)
    v_body_scaled = v_body / v_body[0]
    
    # Extract coordinates on the camera plane
    x_c = v_body_scaled[1]
    y_c = -v_body_scaled[2]
    
    # Convert to pixels
    cx, cy = img_w / 2, img_h / 2
    u = x_c * fx + cx
    v = y_c * fy + cy
    return (u, v)

# --- 3. CAMERA CONFIGURATION ---
IMG_W, IMG_H = 2048, 1536
FX, FY = 1979.2, 1979.2

def run_perspective_shift_simulation():
    print("=== 3D SIMULATION: PERSPECTIVE SHIFT ===\n")
    
    # True physical center of the car (The point we want to find)
    car_center = np.array([0.6, 0.0, 0.0])

    # --- VIEW 1 (Photo 1 taken from the rear of the car) ---
    pos1 = (0.0, 0.0, 0.6)
    att1 = (0.0, -45.0, 0.0)
    
    # YOLO Bounding Box locks onto the rear bumper (10cm offset from center)
    physical_pt1 = np.array([0.5, 0.0, 0.0]) 
    uv1 = world_to_pixel(physical_pt1, pos1, *att1, IMG_W, IMG_H, FX, FY)

    # --- VIEW 2 (Photo 2 taken from the side of the car) ---
    pos2 = (0.6, 1.0, 1.0)
    att2 = (0.0, -45.0, -90.0) 
    
    # YOLO Bounding Box locks onto the side door (10cm offset from center)
    physical_pt2 = np.array([0.6, 0.1, 0.0]) 
    uv2 = world_to_pixel(physical_pt2, pos2, *att2, IMG_W, IMG_H, FX, FY)

    # --- 4. CALL YOUR CALCULATION FUNCTION ---
    print("-> Running triangulate_3d_point from geolocation.py...")
    estimated_point, msg = triangulate_3d_point(
        pos1, att1, uv1,
        pos2, att2, uv2,
        img_w=IMG_W, img_h=IMG_H, fx=FX, fy=FY
    )
    
    if estimated_point is None:
        print("Error:", msg)
        return

    # Calculate Euclidean distance
    error_distance = np.linalg.norm(estimated_point - car_center)

    print(f"\n[CALCULATION RESULTS]")
    print(f"  True Car Center  : [{car_center[0]:.3f}, {car_center[1]:.3f}, {car_center[2]:.3f}]")
    print(f"  Photo 1 Target   : [{physical_pt1[0]:.3f}, {physical_pt1[1]:.3f}, {physical_pt1[2]:.3f}] (Rear Bumper)")
    print(f"  Photo 2 Target   : [{physical_pt2[0]:.3f}, {physical_pt2[1]:.3f}, {physical_pt2[2]:.3f}] (Side Door)")
    print(f"  Estimated Point  : [{estimated_point[0]:.3f}, {estimated_point[1]:.3f}, {estimated_point[2]:.3f}]")
    print(f"\n[ACCURACY]")
    print(f"  Position Error   : {error_distance:.4f} meters")

    # --- 5. 3D PLOT WITH PLOTLY ---
    fig = go.Figure()

    # Draw Ground
    xx = np.linspace(-0.2, 1.2, 10)
    yy = np.linspace(-0.2, 1.2, 10)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.zeros_like(xx)
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.2, colorscale='Greens', showscale=False, name='Ground', hoverinfo='none'))

    # Draw Capture Positions (Photo 1 and 2)
    fig.add_trace(go.Scatter3d(
        x=[pos1[0]], y=[pos1[1]], z=[pos1[2]],
        mode='markers+text', marker=dict(size=8, color='blue', symbol='square'),
        name='Photo 1 (Rear View)', text=[f"Photo 1<br>({pos1[0]:.1f}, {pos1[1]:.1f}, {pos1[2]:.1f})"],
        textposition="top center"
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos2[0]], y=[pos2[1]], z=[pos2[2]],
        mode='markers+text', marker=dict(size=8, color='red', symbol='square'),
        name='Photo 2 (Side View)', text=[f"Photo 2<br>({pos2[0]:.1f}, {pos2[1]:.1f}, {pos2[2]:.1f})"],
        textposition="top center"
    ))

    # Sight ray calculation
    ray_len = 1.8
    v1_world = get_rotation_matrix(*att1) @ np.array([1, (uv1[0]-IMG_W/2)/FX, -(uv1[1]-IMG_H/2)/FY])
    v2_world = get_rotation_matrix(*att2) @ np.array([1, (uv2[0]-IMG_W/2)/FX, -(uv2[1]-IMG_H/2)/FY])
    v1_world = v1_world / np.linalg.norm(v1_world)
    v2_world = v2_world / np.linalg.norm(v2_world)

    # Draw 2 sight rays
    fig.add_trace(go.Scatter3d(
        x=[pos1[0], pos1[0] + v1_world[0]*ray_len], 
        y=[pos1[1], pos1[1] + v1_world[1]*ray_len], 
        z=[pos1[2], pos1[2] + v1_world[2]*ray_len],
        mode='lines', line=dict(color='blue', width=3), name='Ray 1 (Hits Bumper)'
    ))
    fig.add_trace(go.Scatter3d(
        x=[pos2[0], pos2[0] + v2_world[0]*ray_len], 
        y=[pos2[1], pos2[1] + v2_world[1]*ray_len], 
        z=[pos2[2], pos2[2] + v2_world[2]*ray_len],
        mode='lines', line=dict(color='red', width=3), name='Ray 2 (Hits Door)'
    ))

    # Draw Physical Points & Targets
    fig.add_trace(go.Scatter3d(
        x=[car_center[0]], y=[car_center[1]], z=[car_center[2]],
        mode='markers+text', marker=dict(size=10, color='black', symbol='diamond'),
        name='True Car Center', text=[f"Target Center<br>({car_center[0]:.2f}, {car_center[1]:.2f}, {car_center[2]:.2f})"],
        textposition="top right", # Đẩy lên trên và sang phải
        textfont=dict(color='black')
    ))

    fig.add_trace(go.Scatter3d(
        x=[physical_pt1[0]], y=[physical_pt1[1]], z=[physical_pt1[2]],
        mode='markers', marker=dict(size=5, color='blue', symbol='x'), name='Physical Pt 1 (Bumper)'
    ))
    fig.add_trace(go.Scatter3d(
        x=[physical_pt2[0]], y=[physical_pt2[1]], z=[physical_pt2[2]],
        mode='markers', marker=dict(size=5, color='red', symbol='x'), name='Physical Pt 2 (Door)'
    ))

    fig.add_trace(go.Scatter3d(
        x=[estimated_point[0]], y=[estimated_point[1]], z=[estimated_point[2]],
        mode='markers+text', marker=dict(size=8, color='magenta', symbol='circle'),
        name='Estimated Point', text=[f"Est. Point<br>({estimated_point[0]:.2f}, {estimated_point[1]:.2f}, {estimated_point[2]:.2f})"],
        textposition="bottom left", # Đẩy xuống dưới và sang trái
        textfont=dict(color='magenta')
    ))

    # Layout Decorations
    fig.update_layout(
        title=dict(
            text="<b>Perspective Shift Simulation</b><br>Photo 1 vs Photo 2",
            font=dict(size=28)
        ),
        legend=dict(font=dict(size=24)),
        scene=dict(
            xaxis_title='X Axis (meters)',
            yaxis_title='Y Axis (meters)',
            zaxis_title='Z Axis (meters)',
            xaxis=dict(range=[-0.2, 1.2]),
            yaxis=dict(range=[-0.2, 1.2]),
            zaxis=dict(range=[-0.4, 1.2]),
            aspectmode='cube'
        ),
        annotations=[
            dict(
                text=f"<b>Position Error: {error_distance:.4f} m</b>",
                showarrow=False, xref="paper", yref="paper", x=0.02, y=0.95,
                font=dict(color="red", size=20)
            )
        ],
        margin=dict(l=0, r=0, b=0, t=60)
    )

    fig.show()

if __name__ == "__main__":
    run_perspective_shift_simulation()
