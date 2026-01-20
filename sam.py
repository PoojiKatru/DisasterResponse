import os
import cv2
import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from typing import List, TypedDict

class MaskMetadata(TypedDict):
    bbox: List[int] # [x, y, w, h]
    area: float
    predicted_iou: float
    stability_score: float

class SAMMask(TypedDict):
    mask: np.ndarray
    metadata: MaskMetadata

class DroneSAM:
    def __init__(self, checkpoint_path=None, model_type="vit_h"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        possible_paths = [
            checkpoint_path,
            os.path.join(os.getcwd(), "weights", "sam_vit_h_4b8939.pth"),
            os.path.expanduser("~/weights/sam_vit_h_4b8939.pth"),
            "/Users/pkatru/weights/sam_vit_h_4b8939.pth"
        ]
        
        actual_path = None
        for p in possible_paths:
            if p and os.path.exists(p):
                actual_path = p
                break

        if not actual_path:
            raise FileNotFoundError(f"SAM checkpoint not found. Tried: {possible_paths}")

        print(f"Initializing SAM on {self.device} using {actual_path}...")
        self.sam = sam_model_registry[model_type](checkpoint=actual_path)
        self.sam.to(self.device)
        self.sam.eval()
        
        self.points_per_side = 64
        
        # For automatic mask generation (dense)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.85,
            min_mask_region_area=100,
        )
        
        # For prompted mask generation (VLM guided)
        self.predictor = SamPredictor(self.sam)

    def generate_masks_from_prompts(self, image_rgb: np.ndarray, prompts: List[str]) -> List[SAMMask]:
        """
        Generates masks for the scene. 
        Note: This is the slowest part of the pipeline on CPU.
        """
        print(f"Generating geometric masks (Density: {self.points_per_side}^2 points)...")
        print("This may take a few minutes on CPU. Please wait...")
        
        # SAM automatic generator doesn't expose a clean tqdm hook easily, 
        # so we'll just print a start/end message.
        sam_results = self.mask_generator.generate(image_rgb)
        
        print(f"SAM finished. Processing {len(sam_results)} results...")
        
        masks: List[SAMMask] = []
        for res in sam_results:
            mask_obj: SAMMask = {
                "mask": res["segmentation"],
                "metadata": {
                    "bbox": res["bbox"],
                    "area": res["area"],
                    "predicted_iou": res["predicted_iou"],
                    "stability_score": res["stability_score"]
                }
            }
            masks.append(mask_obj)
            
        return masks

    def visualize_masks(self, image_bgr: np.ndarray, masks: List[SAMMask]):
        detections = sv.Detections(
            xyxy=np.array([m["metadata"]["bbox"] for m in masks]), # This needs conversion from [x,y,w,h] to [x1,y1,x2,y2]
            mask=np.array([m["mask"] for m in masks])
        )
        
        # Fix bbox format: [x, y, w, h] -> [x1, y1, x2, y2]
        if len(detections.xyxy) > 0:
            detections.xyxy[:, 2] += detections.xyxy[:, 0]
            detections.xyxy[:, 3] += detections.xyxy[:, 1]

        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        return annotated_image

if __name__ == "__main__":
    # Test script
    image_path = "data/360-6.jpg"
    image_bgr = cv2.imread(image_path)
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_engine = DroneSAM()
        masks = sam_engine.generate_masks_from_prompts(image_rgb, ["survivors", "buildings"])
        print(f"Generated {len(masks)} masks")
        annotated = sam_engine.visualize_masks(image_bgr, masks)
        cv2.imshow("SAM Integration Test", annotated)
        cv2.waitKey(0)
