import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

# Import the core refined modules
from vlm import SatelliteVLM
from sam import DroneSAM
from occupancyplanning import DisasterPlanner

def run_pipeline(image_path, manual_targets=None, start_pos=(50, 50), end_pos=None):
    print(f"\n--- Starting Drone Pipeline: {os.path.basename(image_path)} ---\n")
    
    # 0. Load Image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    h, w = image_bgr.shape[:2]
    
    # Clean up manual targets that are out of bounds
    if manual_targets:
        valid_targets = []
        for r, c in manual_targets:
            if not (0 <= r < h and 0 <= c < w):
                print(f"  [Pipeline Warning]: Removing invalid target {r, c} (Image size is {h}x{w})")
            else:
                valid_targets.append((r, c))
        manual_targets = valid_targets
    
    # Save Step 1: Original
    cv2.imwrite("step1_original.png", image_bgr)
    print("Step 1 Saved: Original Image")
    
    # 1. Run VLM for semantic reasoning
    vlm = SatelliteVLM()
    print("Running VLM semantic analysis...")
    detections = vlm.get_scene_semantics(image_path)
    
    # 2. Run SAM for geometric segmentation
    sam = DroneSAM()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    temp_masks = sam.generate_masks_from_prompts(image_rgb, ["survivors", "damage", "infrastructure"])
    
    # Create the SAM visualization (background for final path)
    sam_vis = sam.visualize_masks(image_bgr, temp_masks)
    
    # 3. Build Occupancy Grid and Plan Path
    # Fuse VLM (roles) and SAM (geometry)
    planner = DisasterPlanner.build_from_perception(
        grid_shape=image_bgr.shape[:2],
        detections=detections,
        masks=temp_masks
    )
    
    # Save Step 2: Occupancy Grid (Black/White)
    planner.save_occupancy_grid("step2_occupancy.png")
    
    # 4. Plan the Mission Path
    # Combine VLM detected targets and manual overrides
    targets = manual_targets if manual_targets else []
    
    # If VLM found targets, add them (if any)
    vlm_targets = [d['metadata']['pos'] for d in detections if d['role'] == 'target' and 'metadata' in d and 'pos' in d['metadata']]
    targets.extend(vlm_targets)
    
    if not targets:
        print("Warning: No targets found. Adding default target for visualization.")
        targets = [(image_bgr.shape[0]-100, image_bgr.shape[1]-100)]
        
    return_base = end_pos if end_pos else start_pos
    full_path = planner.plan_mission(start=start_pos, targets=targets, base=return_base)
    
    # Save Step 3: Final Mission Plot
    planner.visualize_results(
        perception_img=sam_vis,
        full_path=full_path,
        targets=targets,
        save_path="step3_final_path.png",
        start_point=start_pos,
        end_point=return_base
    )
    
    print("\n--- Pipeline Complete! ---")
    print("Outputs generated:")
    print("1. step1_original.png")
    print("2. step2_occupancy.png")
    print("3. step3_final_path.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidated Drone Disaster Pipeline")
    parser.add_argument("image", help="Path to input satellite image")
    parser.add_argument("--targets", nargs="+", type=int, help="Pairs of row col for targets (optional)")
    
    args = parser.parse_args()
    
    manual = []
    if args.targets:
        for i in range(0, len(args.targets), 2):
            manual.append((args.targets[i], args.targets[i+1]))
            
    run_pipeline(args.image, manual_targets=manual)
