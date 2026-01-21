---
title: Drone Command
emoji: 🛰️
colorFrom: purple
colorTo: black
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Drone.Command - Disaster Response Path Planning

An AI-powered drone rescue path planning system that uses satellite imagery analysis to compute safe navigation trajectories through disaster zones.

## Features

- **VLM Analysis**: Uses BLIP to understand scene semantics (roads, buildings, hazards)
- **SAM Segmentation**: Segment Anything Model for precise obstacle detection
- **Path Planning**: A* pathfinding with obstacle avoidance
- **Interactive UI**: Click to select rescue targets and mission waypoints

## Usage

1. Upload a satellite/aerial image
2. Click on the image to mark rescue targets
3. Optionally set start and end points
4. Click "Execute Rescue Pathfinding" to compute the optimal route

## Tech Stack

- Streamlit
- PyTorch
- Transformers (BLIP)
- Segment Anything (SAM)
- OpenCV
