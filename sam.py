import os
import cv2
import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, TypedDict
import requests

class MaskMetadata(TypedDict):
    bbox: List[int]
    area: float
    predicted_iou: float
    stability_score: float

class SAMMask(TypedDict):
    mask: np.ndarray
    metadata: MaskMetadata

class DroneSAM:
    def __init__(self, checkpoint_path=None, model_type="vit_h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        default_path = os.path.join(os.getcwd(), "weights", "sam_vit_h_4b8939.pth")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            actual_path = checkpoint_path
        elif os.path.exists(default_path):
            actual_path = default_path
        else:
            print("SAM checkpoint not found. Downloading...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            response = requests.get(url, stream=True)
            with open(default_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            actual_path = default_path
            print("Download complete.")

        print(f"Initializing SAM on {self.device} using {actual_path}...")
        self.sam = sam_model_registry[model_type](checkpoint=actual_path)
        self.sam.to(self.device)
        self.sam.eval()

        # REDUCED density for speed: 32x32 = 1024 points instead of 64x64 = 4096
        self.points_per_side = 32

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=0.75,  # Higher threshold = fewer masks
            stability_score_thresh=0.88,  # Higher = more stable masks only
            min_mask_region_area=200,  # Larger minimum = fewer tiny masks
        )

        self.predictor = SamPredictor(self.sam)

    def generate_masks_from_prompts(self, image_rgb: np.ndarray, prompts: List[str]) -> List[SAMMask]:
        """
        OPTIMIZED: Reduced point density and stricter filtering
        """
        print(f"Generating SAM masks (Density: {self.points_per_side}x{self.points_per_side} = {self.points_per_side**2} points)...")
        
        sam_results = self.mask_generator.generate(image_rgb)
        
        # FILTER: Keep only high-quality masks
        filtered_results = [
            res for res in sam_results 
            if res['stability_score'] > 0.85 and res['predicted_iou'] > 0.7
        ]
        
        print(f"  Generated {len(sam_results)} masks, filtered to {len(filtered_results)} high-quality masks")

        masks: List[SAMMask] = []
        for res in filtered_results:
            masks.append({
                "mask": res["segmentation"],
                "metadata": {
                    "bbox": res["bbox"],
                    "area": res["area"],
                    "predicted_iou": res["predicted_iou"],
                    "stability_score": res["stability_score"]
                }
            })
        
        print(f"SAM complete: {len(masks)} masks ready for fusion")
        return masks

    def visualize_masks(self, image_bgr: np.ndarray, masks: List[SAMMask]):
        """Visualize SAM segmentation"""
        if not masks:
            return image_bgr
        
        detections = sv.Detections(
            xyxy=np.array([m["metadata"]["bbox"] for m in masks]),
            mask=np.array([m["mask"] for m in masks])
        )
        
        # Convert bbox format [x,y,w,h] -> [x1,y1,x2,y2]
        detections.xyxy[:, 2] += detections.xyxy[:, 0]
        detections.xyxy[:, 3] += detections.xyxy[:, 1]

        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        return annotated_image
