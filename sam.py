import os
import cv2
import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, TypedDict
import requests  # For auto-downloading

# --- TypedDicts for masks ---
class MaskMetadata(TypedDict):
    bbox: List[int]  # [x, y, w, h]
    area: float
    predicted_iou: float
    stability_score: float

class SAMMask(TypedDict):
    mask: np.ndarray
    metadata: MaskMetadata

class DroneSAM:
    def __init__(self, checkpoint_path=None, model_type="vit_h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default checkpoint location
        default_path = os.path.join(os.getcwd(), "weights", "sam_vit_h_4b8939.pth")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)

        # Auto-download if checkpoint is missing
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

        self.points_per_side = 64

        # Automatic mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.85,
            min_mask_region_area=100,
        )

        # Prompted mask generation
        self.predictor = SamPredictor(self.sam)

    def generate_masks_from_prompts(self, image_rgb: np.ndarray, prompts: List[str]) -> List[SAMMask]:
        print(f"Generating geometric masks (Density: {self.points_per_side}^2 points)...")
        sam_results = self.mask_generator.generate(image_rgb)

        masks: List[SAMMask] = []
        for res in sam_results:
            masks.append({
                "mask": res["segmentation"],
                "metadata": {
                    "bbox": res["bbox"],
                    "area": res["area"],
                    "predicted_iou": res["predicted_iou"],
                    "stability_score": res["stability_score"]
                }
            })
        print(f"SAM finished. Processed {len(masks)} masks.")
        return masks

    def visualize_masks(self, image_bgr: np.ndarray, masks: List[SAMMask]):
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

if __name__ == "__main__":
    # Test script for local use
    image_path = "data/360-6.jpg"
    image_bgr = cv2.imread(image_path)
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_engine = DroneSAM()
        masks = sam_engine.generate_masks_from_prompts(image_rgb, ["survivors", "buildings"])
        print(f"Generated {len(masks)} masks")

        # Streamlit-compatible display (replace cv2.imshow)
        from PIL import Image
        import streamlit as st
        annotated = sam_engine.visualize_masks(image_bgr, masks)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="SAM Segmentation", use_column_width=True)
