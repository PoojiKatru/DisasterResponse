import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import time

# Import optimized modules
from vlm import SatelliteVLM
from sam import DroneSAM
from occupancyplanning import DisasterPlanner

def run_pipeline(image_path, manual_targets=None, start_pos=(50, 50), end_pos=None):
    """
    OPTIMIZED PIPELINE:
    - Fast VLM analysis (3 calls instead of 9+)
    - Reduced SAM density (32x32 instead of 64x64)
    - Timeout-protected A* pathfinding
    - Actual VLM-SAM semantic fusion
    """
    print(f"\n{'='*70}")
    print(f"DRONE RESCUE PIPELINE: {os.path.basename(image_path)}")
    print(f"{'='*70}\n")
    
    overall_start = time.time()
    
    # 0. Load Image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"ERROR: Could not load image at {image_path}")
        return
    
    h, w = image_bgr.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Validate targets
    if manual_targets:
        valid_targets = []
        for r, c in manual_targets:
            if not (0 <= r < h and 0 <= c < w):
                print(f"  [WARNING] Removing invalid target ({r}, {c}) - out of bounds")
            else:
                valid_targets.append((r, c))
        manual_targets = valid_targets
        print(f"Valid targets: {len(manual_targets)}")
    
    # Save Step 1: Original
    cv2.imwrite("step1_original.png", image_bgr)
    print(f"✓ Step 1 saved: step1_original.png\n")
    
    # 1. VLM Analysis (OPTIMIZED: 3 calls instead of 9+)
    print("STAGE 1: VLM SEMANTIC ANALYSIS")
    print("-" * 70)
    vlm_start = time.time()
    vlm = SatelliteVLM()
    detections = vlm.get_scene_semantics(image_path)
    vlm_time = time.time() - vlm_start
    print(f"✓ VLM complete in {vlm_time:.1f}s")
    print(f"  Detected categories:")
    for det in detections:
        print(f"    - {det['label']}: {det['role']} (priority {det['priority']})")
    print()
    
    # 2. SAM Segmentation (OPTIMIZED: 32x32 points, filtered masks)
    print("STAGE 2: SAM GEOMETRIC SEGMENTATION")
    print("-" * 70)
    sam_start = time.time()
    sam = DroneSAM()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = sam.generate_masks_from_prompts(image_rgb, ["scene"])
    sam_vis = sam.visualize_masks(image_bgr, masks)
    sam_time = time.time() - sam_start
    print(f"✓ SAM complete in {sam_time:.1f}s\n")
    
    # 3. Perception Fusion (IMPROVED: VLM semantics guide SAM interpretation)
    print("STAGE 3: VLM-SAM FUSION")
    print("-" * 70)
    fusion_start = time.time()
    planner = DisasterPlanner.build_from_perception(
        grid_shape=image_bgr.shape[:2],
        detections=detections,
        masks=masks
    )
    fusion_time = time.time() - fusion_start
    print(f"✓ Fusion complete in {fusion_time:.1f}s\n")
    
    # Save Step 2: Occupancy Grid
    planner.save_occupancy_grid("step2_occupancy.png")
    print(f"✓ Step 2 saved: step2_occupancy.png\n")
    
    # 4. Path Planning (OPTIMIZED: Timeout protection, greedy TSP)
    print("STAGE 4: PATH PLANNING")
    print("-" * 70)
    planning_start = time.time()
    
    # Prepare targets
    targets = manual_targets if manual_targets else []
    
    # Add VLM-detected targets if any
    vlm_targets = [
        d['metadata']['pos'] for d in detections 
        if d['role'] == 'target' and 'metadata' in d and 'pos' in d['metadata']
    ]
    targets.extend(vlm_targets)
    
    if not targets:
        print("  No targets specified. Adding default target for demo.")
        targets = [(h - 100, w - 100)]
    
    print(f"Planning for {len(targets)} targets...")
    
    return_base = end_pos if end_pos else start_pos
    full_path = planner.plan_mission(start=start_pos, targets=targets, base=return_base)
    planning_time = time.time() - planning_start
    print(f"✓ Path planning complete in {planning_time:.1f}s\n")
    
    # Save Step 3: Final Visualization
    print("STAGE 5: VISUALIZATION")
    print("-" * 70)
    planner.visualize_results(
        perception_img=sam_vis,
        full_path=full_path,
        targets=targets,
        save_path="step3_final_path.png",
        start_point=start_pos,
        end_point=return_base
    )
    print(f"✓ Step 3 saved: step3_final_path.png\n")
    
    # Summary
    overall_time = time.time() - overall_start
    print(f"{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {overall_time:.1f}s")
    print(f"  VLM:      {vlm_time:.1f}s ({vlm_time/overall_time*100:.1f}%)")
    print(f"  SAM:      {sam_time:.1f}s ({sam_time/overall_time*100:.1f}%)")
    print(f"  Fusion:   {fusion_time:.1f}s ({fusion_time/overall_time*100:.1f}%)")
    print(f"  Planning: {planning_time:.1f}s ({planning_time/overall_time*100:.1f}%)")
    print(f"\nOutputs generated:")
    print(f"  1. step1_original.png")
    print(f"  2. step2_occupancy.png")
    print(f"  3. step3_final_path.png")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Drone Disaster Pipeline")
    parser.add_argument("image", help="Path to input satellite image")
    parser.add_argument("--targets", nargs="+", type=int, help="Pairs of row col for targets")
    
    args = parser.parse_args()
    
    manual = []
    if args.targets:
        for i in range(0, len(args.targets), 2):
            manual.append((args.targets[i], args.targets[i+1]))
    
    run_pipeline(args.image, manual_targets=manual)
