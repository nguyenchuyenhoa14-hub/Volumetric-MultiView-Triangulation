# Volumetric Multi-View Triangulation for UAV 3D Target Geolocation

This repository contains the Python implementation of a terrain-independent volumetric multi-view triangulation algorithm for UAV 3D target geolocation.

## Overview

This project implements a perspective-robust, terrain-independent method for 3D target geolocation using a moving monocular UAV. 

By capturing images of a target from two sequential viewpoints, the system:
1. Projects a 5-point bounding box (representing the target's volumetric frustum) from camera coordinate frames into 3D world space.
2. Performs skew-line triangulation across the viewing rays.
3. Computes the 3D target centroid.

## Directory Structure

```
├── src/
│   ├── __init__.py
│   └── geolocation.py        # Core triangulation & sight-vector math
├── tests/
│   ├── __init__.py
│   ├── simulate_3d.py        # 3D visualization of single-ray projection
│   ├── simulate_skew_3d.py   # 3D visualization of skew-line intersection
│   ├── simulate_frustum.py   # 3D visualization of volumetric frustum intersection
│   ├── test_scenarios.py     # Simple test cases for coordinate verification
│   └── test_stereo_*.py      # Integration testing with simulated YOLO outputs
├── main.py                   # Main simulation runner
└── requirements.txt          # Dependencies
```

## Setup & Installation

Ensure you have Python 3.8+ and the required packages installed:

```bash
pip install -r requirements.txt
```

### Running Simulations

To run the standard verification scenarios:

```bash
python main.py
```

To run 3D visual simulations (requires matplotlib):

```bash
python -m tests.simulate_3d
python -m tests.simulate_skew_3d
python -m tests.simulate_frustum_centroid
```
