# UAV Volumetric 3D Geolocation

A terrain-independent volumetric multi-view triangulation system for UAV-based 3D target geolocation.

## Overview

This project implements a multi-view 3D geolocation algorithm estimating physical coordinates of ground targets from sequential aerial images. By combining camera telemetry (GPS/IMU) with intrinsic camera matrices, the system projects viewing rays and calculates target coordinates without requiring Digital Elevation Models (DEM).

### Key Results
- **Accuracy Improvement:** Reduces target localization error by **22.6%** compared to single-ray projection baselines.
- **Terrain Independence:** Evaluates volumetric ray intersections rather than assuming flat ground, enabling operation on irregular terrain.
- **Publication Status:** Accepted for presentation at **IAAA 2026**.

## Algorithmic Methodology

### 1. Multi-Ray Frustum Intersection
Rather than projecting a single ray from the camera center through a target pixel, the algorithm models projection boundaries as volumetric frustums to account for telemetry noise in drone IMU/GPS sensors.

### 2. Skew-Line Triangulation
Due to telemetry noise, rays projected from different drone positions do not intersect perfectly in 3D space. The algorithm:
- Computes the shortest line segment connecting skew projection lines.
- Estimates target coordinates as the spatial midpoint of this segment.
- Applies weighted least-squares optimization across sequential frames to filter outliers.

## Repository Structure

- `src/`: Core Python implementation of triangulation and frustum intersection algorithms.
- `sim/`: Simulation scripts generating synthetic UAV telemetry and target projection coordinates.
- `data/`: Sample telemetry and pixel coordinates (KIIT-MiTA dataset).
- `evaluation/`: Accuracy evaluation scripts comparing multi-ray results against single-ray baselines.
