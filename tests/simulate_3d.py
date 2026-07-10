import sys
import os
import numpy as np
import math
import plotly.graph_objects as go

# --- 1. CORE MATH & PHYSICS FUNCTIONS ---
def get_rotation_matrix(roll, pitch, yaw):
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return R_z @ R_y @ R_x

def pixel_to_ray(u, v, roll, pitch, yaw, img_w, img_h, fx, fy):
    """Converts a 2D pixel coordinate back into a 3D directional ray vector."""
    cx, cy = img_w / 2, img_h / 2
    x_c = (u - cx) / fx
    y_c = (v - cy) / fy
    
    # Vector in Camera Frame (Forward is X, Right is Y, Down is Z)
    v_body = np.array([1.0, x_c, -y_c]) 
    
    # Rotate to World Frame
    R = get_rotation_matrix(roll, pitch, yaw)
    v_world = R @ v_body
    return v_world / np.linalg.norm(v_world) # Normalize to unit vector

# --- 2. CONFIGURATION ---
IMG_W, IMG_H = 2048, 1536
FX, FY = 1979.2, 1979.2

# Camera 1 Setup
pos1 = np.array([0.0, 0.0, 0.6])
att1 = (0.0, -45.0, 0.0)

# Camera 2 Setup
pos2 = np.array([0.6, 1.0, 1.0])
att2 = (0.0, -45.0, -90.0)

# Target Setup (True Car Center)
car_center = np.array([0.6, 0.0, 0.0])

def run_frustum_simulation():
    print("=== 3D SIMULATION: UNCERTAINTY FRUSTUM ===")
    
    # --- 3. MOCKING YOLO BOUNDING BOXES ---
    # In a real scenario, YOLO gives you [u_min, v_min, u_max, v_max]
    # Here we define mock bounding boxes that simulate seeing a car (roughly 400x300 pixels on image)
    
    # Photo 1 Bounding Box (Rear View)
    center_u1, center_v1 = 1024, 768
    w1, h1 = 500, 300
    bbox1 = {
        'tl': (center_u1 - w1/2, center_v1 - h1/2), # Top-Left
        'tr': (center_u1 + w1/2, center_v1 - h1/2), # Top-Right
        'br': (center_u1 + w1/2, center_v1 + h1/2), # Bottom-Right
        'bl': (center_u1 - w1/2, center_v1 + h1/2)  # Bottom-Left
    }

    # Photo 2 Bounding Box (Side View - wider because car is longer from the side)
    center_u2, center_v2 = 1024, 768
    w2, h2 = 700, 250
    bbox2 = {
        'tl': (center_u2 - w2/2, center_v2 - h2/2),
        'tr': (center_u2 + w2/2, center_v2 - h2/2),
        'br': (center_u2 + w2/2, center_v2 + h2/2),
        'bl': (center_u2 - w2/2, center_v2 + h2/2)
    }

    # --- 4. GENERATE FRUSTUM GEOMETRY ---
    ray_len = 1.5 # Length of the frustum to draw (meters)
    
    def get_frustum_vertices(pos, att, bbox, length):
        v_tl = pos + pixel_to_ray(*bbox['tl'], *att, IMG_W, IMG_H, FX, FY) * length
        v_tr = pos + pixel_to_ray(*bbox['tr'], *att, IMG_W, IMG_H, FX, FY) * length
        v_br = pos + pixel_to_ray(*bbox['br'], *att, IMG_W, IMG_H, FX, FY) * length
        v_bl = pos + pixel_to_ray(*bbox['bl'], *att, IMG_W, IMG_H, FX, FY) * length
        # Returns 5 points: Camera Origin + 4 Far Corners
        return np.vstack([pos, v_tl, v_tr, v_br, v_bl])

    vertices_1 = get_frustum_vertices(pos1, att1, bbox1, ray_len)
    vertices_2 = get_frustum_vertices(pos2, att2, bbox2, ray_len)

    # --- 5. PLOTTING WITH PLOTLY ---
    fig = go.Figure()

    # Draw Ground
    xx = np.linspace(-0.2, 1.2, 10)
    yy = np.linspace(-0.2, 1.2, 10)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.zeros_like(xx)
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, opacity=0.1, colorscale='Greens', showscale=False, hoverinfo='none'))

    # Helper function to draw 3D Pyramids (Frustums)
    def add_frustum(fig, vertices, color, name):
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        # Faces of the pyramid defined by vertex indices
        i = [0, 0, 0, 0, 1, 1]
        j = [1, 2, 3, 4, 2, 3]
        k = [2, 3, 4, 1, 3, 4]
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color=color, opacity=0.15, name=name, showlegend=True
        ))
        
        # Draw edges of the frustum
        edges = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[x[edge[0]], x[edge[1]]], y=[y[edge[0]], y[edge[1]]], z=[z[edge[0]], z[edge[1]]],
                mode='lines', line=dict(color=color, width=2), showlegend=False
            ))

    # Add Frustum 1 (Blue) and Frustum 2 (Red)
    add_frustum(fig, vertices_1, 'blue', 'Frustum 1 (Rear Bounding Box)')
    add_frustum(fig, vertices_2, 'red', 'Frustum 2 (Side Bounding Box)')

    # Draw Cameras
    fig.add_trace(go.Scatter3d(x=[pos1[0]], y=[pos1[1]], z=[pos1[2]], mode='markers+text', marker=dict(size=8, color='blue', symbol='square'), name='Photo 1', text=['Photo 1'], textposition='top center'))
    fig.add_trace(go.Scatter3d(x=[pos2[0]], y=[pos2[1]], z=[pos2[2]], mode='markers+text', marker=dict(size=8, color='red', symbol='square'), name='Photo 2', text=['Photo 2'], textposition='top center'))

    # Draw True Car Target (Center of the intersection)
    fig.add_trace(go.Scatter3d(
        x=[car_center[0]], y=[car_center[1]], z=[car_center[2]],
        mode='markers+text', marker=dict(size=12, color='black', symbol='diamond'),
        name='True Target', text=['True Target Center'], textposition='bottom center'
    ))

    # Layout Setup
    fig.update_layout(
        title="<b>Frustum Intersection (Uncertainty Volume)</b><br>Using Bounding Box corners instead of Center Point",
        scene=dict(
            xaxis_title='X Axis (meters)', yaxis_title='Y Axis (meters)', zaxis_title='Z Axis (meters)',
            xaxis=dict(range=[-0.2, 1.2]), yaxis=dict(range=[-0.2, 1.2]), zaxis=dict(range=[-0.4, 1.2]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(x=0.02, y=0.98)
    )

    fig.show()

if __name__ == "__main__":
    run_frustum_simulation()