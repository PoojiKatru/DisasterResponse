import os
import cv2
import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, TypedDict
import requests

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

class MaskMetadata(TypedDict):
    bbox: List[int]
    area: float
    predicted_iou: float
    stability_score: float

class SAMMask(TypedDict):
    mask: np.ndarray
    metadata: MaskMetadata


def _load_sam_model(model_type="vit_h"):
    """Load SAM model (cached by Streamlit if available)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_path = os.path.join(os.getcwd(), "weights", "sam_vit_h_4b8939.pth")
    os.makedirs(os.path.dirname(default_path), exist_ok=True)

    if os.path.exists(default_path):
        actual_path = default_path
    else:
        print("SAM checkpoint not found. Downloading (~2.4GB)...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(default_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\rDownloading SAM: {pct:.1f}%", end="", flush=True)
        print("\nDownload complete.")
        actual_path = default_path

    print(f"Loading SAM model on {device}...")
    sam = sam_model_registry[model_type](checkpoint=actual_path)
    sam.to(device)
    sam.eval()
    print("SAM model loaded!")
    return sam, device


# Apply Streamlit caching if available
if HAS_STREAMLIT:
    _load_sam_model = st.cache_resource(show_spinner="Loading SAM model... (this may take a few minutes on first run)")(_load_sam_model)


class DroneSAM:
    def __init__(self, checkpoint_path=None, model_type="vit_h"):
        # Use cached model loading
        self.sam, self.device = _load_sam_model(model_type)

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
