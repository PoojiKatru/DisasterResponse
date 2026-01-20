"""
Optimized Satellite Image Analysis VLM
Fixed: Removed slow multi-inference loops, streamlined for speed
"""

import torch
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from typing import List, TypedDict, Optional

class DetectedObject(TypedDict):
    label: str
    role: str  # 'target', 'obstacle', 'hazard', 'clear'
    priority: int  # 10 (highest) to 1 (lowest)
    metadata: Optional[dict]

class SatelliteVLM:
    """
    Optimized VLM - Single-pass analysis instead of 9+ inference calls
    """
    
    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Initializing VLM on {self.device}...")
        
        try:
            # Use base models for speed
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            
            self.qa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.qa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(self.device)
            
            print("VLM ready!\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def caption_image(self, image_path, prompt=None):
        """Generate a caption describing the image"""
        image = Image.open(image_path).convert('RGB')
        
        if prompt:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_length=50)
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    
    def answer_question(self, image_path, question):
        """Answer a question about the image"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.qa_processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.qa_model.generate(**inputs, max_length=20)
        
        answer = self.qa_processor.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    def get_scene_semantics(self, image_path: str) -> List[DetectedObject]:
        """
        OPTIMIZED: Fast single-pass analysis for disaster response
        Instead of 9+ calls, we do 3 targeted questions
        """
        print("Running fast VLM analysis (3 inference calls)...")
        
        detected_objects: List[DetectedObject] = []
        
        # 1. General scene understanding (1 call)
        scene_type = self.caption_image(image_path, prompt="an aerial view of")
        print(f"  Scene: {scene_type}")
        
        # 2. Obstacle detection (1 call)
        obstacles = self.answer_question(image_path, "What obstacles are visible?")
        print(f"  Obstacles: {obstacles}")
        
        # 3. Hazard detection (1 call)  
        hazards = self.answer_question(image_path, "Are there any hazards or dangers?")
        print(f"  Hazards: {hazards}")
        
        # Parse responses into structured data
        scene_lower = scene_type.lower()
        obstacles_lower = obstacles.lower()
        hazards_lower = hazards.lower()
        
        # Detect ROADS (navigable areas - mark as clear)
        if any(word in scene_lower for word in ["road", "street", "highway", "pavement"]):
            detected_objects.append({
                "label": "road",
                "role": "clear",
                "priority": 10,
                "metadata": {"description": "navigable road surface"}
            })
        
        # Detect BUILDINGS (obstacles)
        if any(word in obstacles_lower for word in ["building", "house", "structure", "roof"]):
            detected_objects.append({
                "label": "building",
                "role": "obstacle",
                "priority": 8,
                "metadata": {"description": obstacles}
            })
        
        # Detect VEGETATION (obstacles)
        if any(word in obstacles_lower or word in scene_lower for word in ["tree", "forest", "vegetation", "bush"]):
            detected_objects.append({
                "label": "vegetation",
                "role": "obstacle", 
                "priority": 6,
                "metadata": {"description": "dense vegetation"}
            })
        
        # Detect WATER (hazard)
        if any(word in hazards_lower or word in scene_lower for word in ["water", "flood", "river", "lake"]):
            detected_objects.append({
                "label": "water",
                "role": "hazard",
                "priority": 9,
                "metadata": {"description": "water hazard"}
            })
        
        # Detect DAMAGE/DISASTER
        if any(word in scene_lower or word in hazards_lower for word in ["damage", "destroyed", "debris", "disaster"]):
            detected_objects.append({
                "label": "damage",
                "role": "hazard",
                "priority": 7,
                "metadata": {"description": "damaged area"}
            })
        
        # Default: assume general terrain if nothing detected
        if not detected_objects:
            detected_objects.append({
                "label": "terrain",
                "role": "clear",
                "priority": 1,
                "metadata": {"description": "general terrain"}
            })
        
        print(f"VLM detected {len(detected_objects)} semantic categories")
        return detected_objects


if __name__ == "__main__":
    print("Testing Optimized VLM...")
    vlm = SatelliteVLM()
    
    image_path = input("Enter image path: ").strip()
    semantics = vlm.get_scene_semantics(image_path)
    
    print("\n=== DETECTED SEMANTICS ===")
    for obj in semantics:
        print(f"- {obj['label']}: {obj['role']} (priority {obj['priority']})")
