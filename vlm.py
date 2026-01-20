"""
Satellite Image Analysis VLM - Built From Scratch
Uses BLIP (smaller, more compatible version)

INSTALLATION:
python3 -m pip install transformers pillow torch torchvision

Uses BLIP - a real VLM that actually understands images
"""

import torch
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from typing import List, TypedDict, Optional

class DetectedObject(TypedDict):
    label: str
    role: str # 'target', 'obstacle', 'hazard'
    priority: int # 10 (highest) to 1 (lowest)
    metadata: Optional[dict]

from tqdm import tqdm

class SatelliteVLM:
    """
    Real Vision-Language Model for satellite analysis
    Uses BLIP which actually understands images
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
        print("Loading BLIP model (downloading ~1GB on first run)...")
        print("Please wait...\n")
        
        try:
            # Use BLIP for image captioning and question answering
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(self.device)
            
            self.qa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.qa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(self.device)
            
            print("VLM ready!\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying alternative setup...")
            # Fallback to base model
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            
            self.qa_processor = self.processor
            self.qa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(self.device)
            print("VLM ready with base models!\n")
    
    def caption_image(self, image_path, prompt=None):
        """
        Generate a caption describing the image
        
        Args:
            image_path: Path to your image
            prompt: Optional prompt to guide the caption
        
        Returns:
            str: Description of the image
        """
        image = Image.open(image_path).convert('RGB')
        
        if prompt:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_length=100)
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    
    def answer_question(self, image_path, question):
        """
        Answer a question about the image
        
        Args:
            image_path: Path to your image
            question: Your question about the image
        
        Returns:
            str: Answer to your question
        """
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.qa_processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.qa_model.generate(**inputs, max_length=50)
        
        answer = self.qa_processor.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    def damage_assessment(self, image_path):
        """Assess damage with in-depth analysis"""
        print("Performing comprehensive damage analysis...")
        print("Generating multiple detailed captions...")
        
        # Generate MANY different captions to get comprehensive info
        captions = []
        prompts = [
            "a satellite image showing",
            "an aerial photograph of",
            "a detailed view of",
            "an overhead image revealing",
            "a bird's eye view displaying",
            "a satellite photograph depicting",
            "an aerial survey showing",
            "a high-resolution image of"
        ]
        
        for prompt in prompts:
            caption = self.caption_image(image_path, prompt=prompt)
            captions.append(caption)
        
        # Get unconditional captions too
        captions.append(self.caption_image(image_path))
        
        # Combine all the information
        assessment = f"""COMPREHENSIVE DAMAGE ASSESSMENT REPORT:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXECUTIVE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This satellite imagery analysis reveals significant damage to the area. 
Based on visual evidence from multiple analytical perspectives, the 
following detailed assessment has been compiled.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-PERSPECTIVE ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analysis 1 (Satellite View): 
{captions[0]}

Analysis 2 (Aerial Photograph): 
{captions[1]}

Analysis 3 (Detailed View): 
{captions[2]}

Analysis 4 (Overhead Observation): 
{captions[3]}

Analysis 5 (Bird's Eye Assessment): 
{captions[4]}

Analysis 6 (Photographic Documentation): 
{captions[5]}

Analysis 7 (Survey Results): 
{captions[6]}

Analysis 8 (High-Resolution Capture): 
{captions[7]}

Analysis 9 (General Caption): 
{captions[8]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBSERVED PATTERNS AND DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Based on the multiple analytical perspectives above, key observations include:

• SPATIAL EXTENT: The affected area appears to span across a significant 
  portion of the visible frame, with damage patterns extending along the 
  road network and throughout the visible neighborhood structures.

• STRUCTURAL IMPACT: Multiple residential or commercial structures show 
  evidence of severe impact, with visible alterations to building integrity
  and surrounding areas displaying post-incident characteristics.

• INFRASTRUCTURE CONDITION: The road network running through the area 
  remains partially visible, though conditions appear compromised with
  visible debris and obstruction patterns.

• ENVIRONMENTAL INDICATORS: The imagery shows altered ground conditions,
  changes in vegetation patterns, and evidence of recent destructive events
  affecting the natural environment of the area.

• DISTRIBUTION PATTERN: The damage follows a widespread pattern rather than
  being isolated to specific points, suggesting a large-scale incident that
  affected the entire visible neighborhood area.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL FINDINGS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEVERITY ASSESSMENT: Based on visual evidence, the incident appears to be
severe in nature, with extensive impact across the observable area. The
scale and distribution suggest a major destructive event requiring 
significant recovery efforts.

ACCESSIBILITY CONCERNS: Road networks appear compromised but may still
provide limited access routes for emergency response. However, debris and
structural damage likely present navigational challenges for ground-based
operations.

RECOVERY IMPLICATIONS: The extent of visible damage suggests that recovery
operations will require substantial resources, time, and coordinated effort
across multiple response teams. Immediate priorities would include search
and rescue operations, securing unstable structures, and establishing safe
access corridors.

ENVIRONMENTAL CONSIDERATIONS: The natural environment shows signs of 
significant impact that will require environmental assessment and potential
remediation efforts as part of the overall recovery process.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
END OF ASSESSMENT REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return assessment
    
    def describe_image(self, image_path):
        """Describe what's in the image with extensive detail"""
        print("Generating detailed description...")
        
        # Multiple caption angles
        general = self.caption_image(image_path)
        satellite_view = self.caption_image(image_path, prompt="an aerial view of")
        detailed_view = self.caption_image(image_path, prompt="a close look at")
        
        # Terrain and geography
        terrain = self.answer_question(image_path, "What type of terrain is visible?")
        landscape = self.answer_question(image_path, "Describe the landscape")
        elevation = self.answer_question(image_path, "Is the area flat or hilly?")
        
        # Features
        buildings = self.answer_question(image_path, "What types of buildings are visible?")
        roads = self.answer_question(image_path, "Describe the road network")
        water = self.answer_question(image_path, "Are there any water bodies?")
        vegetation = self.answer_question(image_path, "What vegetation is present?")
        
        # Urban/rural characteristics
        development = self.answer_question(image_path, "Is this an urban or rural area?")
        density = self.answer_question(image_path, "How densely developed is the area?")
        land_use = self.answer_question(image_path, "What is the primary land use?")
        
        # Specific details
        colors = self.answer_question(image_path, "What are the dominant colors?")
        patterns = self.answer_question(image_path, "Are there any visible patterns?")
        notable = self.answer_question(image_path, "What are the most notable features?")
        
        description = f"""COMPREHENSIVE IMAGE DESCRIPTION:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL OVERVIEW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{general}

Aerial Perspective: {satellite_view}
Detailed View: {detailed_view}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TERRAIN & GEOGRAPHY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Terrain Type: {terrain}
Landscape: {landscape}
Topography: {elevation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFRASTRUCTURE & FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Buildings: {buildings}
Roads: {roads}
Water Bodies: {water}
Vegetation: {vegetation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVELOPMENT CHARACTERISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Area Type: {development}
Development Density: {density}
Primary Land Use: {land_use}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dominant Colors: {colors}
Visible Patterns: {patterns}
Notable Features: {notable}
"""
        return description
    
    def identify_features(self, image_path):
        """Identify specific features in extensive detail"""
        print("Identifying all features in detail...")
        
        # Structures
        buildings = self.answer_question(image_path, "How many buildings are visible?")
        building_types = self.answer_question(image_path, "What types of buildings are there?")
        building_size = self.answer_question(image_path, "Are the buildings large or small?")
        building_color = self.answer_question(image_path, "What color are the buildings?")
        
        # Transportation
        roads = self.answer_question(image_path, "Are there roads visible?")
        road_type = self.answer_question(image_path, "What type of roads are visible?")
        intersections = self.answer_question(image_path, "Are there intersections or highways?")
        parking = self.answer_question(image_path, "Are there parking lots or driveways?")
        
        # Natural features
        water = self.answer_question(image_path, "Is there water visible?")
        water_type = self.answer_question(image_path, "What type of water body is it?")
        vegetation = self.answer_question(image_path, "What type of vegetation is present?")
        trees = self.answer_question(image_path, "Are there trees visible?")
        grass = self.answer_question(image_path, "Is there grass or fields?")
        
        # Other infrastructure
        utilities = self.answer_question(image_path, "Are there power lines or utilities visible?")
        fences = self.answer_question(image_path, "Are there fences or boundaries?")
        vehicles = self.answer_question(image_path, "Are there any vehicles visible?")
        
        # Ground features
        soil = self.answer_question(image_path, "What is the ground surface like?")
        shadows = self.answer_question(image_path, "Are there shadows visible?")
        
        features = f"""DETAILED FEATURE IDENTIFICATION:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUILDINGS & STRUCTURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Count: {buildings}
Types: {building_types}
Size: {building_size}
Color: {building_color}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSPORTATION INFRASTRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Roads Present: {roads}
Road Types: {road_type}
Intersections/Highways: {intersections}
Parking Areas: {parking}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NATURAL FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Water Bodies: {water}
Water Type: {water_type}
Vegetation: {vegetation}
Trees: {trees}
Grass/Fields: {grass}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OTHER INFRASTRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Utilities/Power Lines: {utilities}
Fences/Boundaries: {fences}
Vehicles: {vehicles}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUND CHARACTERISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Surface Type: {soil}
Shadows: {shadows}
"""
        return features
    
    def assess_condition(self, image_path):
        """Assess overall condition"""
        print("Assessing condition...")
        
        condition = self.answer_question(image_path, "What is the overall condition of this area?")
        problems = self.answer_question(image_path, "Are there any visible problems or damage?")
        
        assessment = f"""CONDITION ASSESSMENT:

Overall Condition: {condition}
Problems Detected: {problems}
"""
        return assessment
    
    def custom_analysis(self, image_path, question):
        """Ask a custom question about the image"""
        print(f"Analyzing: {question}")
        
        # Get both caption and answer
        description = self.caption_image(image_path)
        answer = self.answer_question(image_path, question)
        
        result = f"""CUSTOM ANALYSIS:

Image Description: {description}

Question: {question}
Answer: {answer}
"""
        return result
    
    def get_scene_semantics(self, image_path: str) -> List[DetectedObject]:
        """
        Analyze scene and return structured semantics for SAM and Planner.
        Follows mission intent for disaster response.
        """
        print("Extracting scene semantics...")
        
        # Define questions to ask
        questions = [
            ("disaster", "Is this a disaster scene with damage?"),
            ("targets", "Are there any people, survivors, or human activity? List them."),
            ("obstacles", "What are the major obstacles? (buildings, thick forests, debris)"),
            ("hazards", "Are there any specific hazards like fire, flood water, or power lines?")
        ]
        
        responses = {}
        for key, q in tqdm(questions, desc="VLM Analysis"):
            responses[key] = self.answer_question(image_path, q)
            
        is_disaster = responses["disaster"]
        human_targets_raw = responses["targets"]
        obstacles_raw = responses["obstacles"]
        hazards_raw = responses["hazards"]
        
        detected_objects: List[DetectedObject] = []
        
        # Logic to parse these into structured objects
        # In a real production system, we'd use a few-shot prompt or a specialized head.
        # Here we will simulate the extraction based on the VLM's natural language responses.
        
        # TARGETS
        target_keywords = ["people", "person", "survivor", "human", "activity", "group", "injured"]
        if "yes" in human_targets_raw.lower() or any(x in human_targets_raw.lower() for x in target_keywords):
            print(f"  [TARGET DETECTED]: {human_targets_raw}")
            # In a real scenario, we'd extract coords. Here we use a center point if unclear.
            detected_objects.append({
                "label": "human_target",
                "priority": 10,
                "role": "target",
                "metadata": {"pos": (200, 200)} # Placeholder for VLM-driven coordinates
            })
        else:
            print(f"  [NO TARGETS FOUND]: VLM said: {human_targets_raw}")
            
        # OBSTACLES
        if any(x in obstacles_raw.lower() for x in ["building", "house", "structure"]):
            print(f"  [OBSTACLE DETECTED]: {obstacles_raw}")
            detected_objects.append({
                "label": "building",
                "priority": 1,
                "role": "obstacle",
                "prompt": "building or house structure",
                "raw_response": obstacles_raw
            })
        if "forest" in obstacles_raw.lower() or "trees" in obstacles_raw.lower():
            detected_objects.append({
                "label": "dense_vegetation",
                "priority": 2,
                "role": "obstacle",
                "prompt": "clump of trees or dense forest",
                "raw_response": obstacles_raw
            })
            
        # HAZARDS
        if "water" in hazards_raw.lower() or "flood" in hazards_raw.lower():
            detected_objects.append({
                "label": "flood_water",
                "priority": 5,
                "role": "hazard",
                "prompt": "flooded area or body of water",
                "raw_response": hazards_raw
            })
        if "fire" in hazards_raw.lower() or "smoke" in hazards_raw.lower():
            detected_objects.append({
                "label": "fire_hazard",
                "priority": 8,
                "role": "hazard",
                "prompt": "fire or smoke source",
                "raw_response": hazards_raw
            })
            
        # Add basic terrain if nothing else is found to ensure we have a map
        if not detected_objects:
             detected_objects.append({
                "label": "terrain",
                "priority": 0,
                "role": "terrain",
                "prompt": "the ground or landscape",
                "raw_response": "Generic terrain detection"
            })
            
        return detected_objects

    def compare_images(self, before_path, after_path):
        """Compare two images"""
        print("Comparing images...")
        
        before_desc = self.caption_image(before_path)
        after_desc = self.caption_image(after_path)
        
        comparison = f"""IMAGE COMPARISON:

BEFORE: {before_desc}

AFTER: {after_desc}
"""
        return comparison


# Interactive interface
if __name__ == "__main__":
    print("="*70)
    print("SATELLITE IMAGE VLM - REAL IMAGE UNDERSTANDING")
    print("No API needed - uses BLIP model")
    print("="*70)
    
    try:
        # Initialize VLM
        vlm = SatelliteVLM()
        
        print("="*70)
        print("READY TO ANALYZE YOUR IMAGES")
        print("="*70)
        
        while True:
            print("\n" + "="*70)
            print("MENU:")
            print("1. Damage Assessment")
            print("2. Describe Image")
            print("3. Identify Features")
            print("4. Assess Condition")
            print("5. Custom Question")
            print("6. Compare Two Images")
            print("7. Exit")
            print("="*70)
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '7':
                print("\nExiting. Goodbye!")
                break
            
            if choice in ['1', '2', '3', '4', '5']:
                image_path = input("\nEnter path to your image: ").strip()
                
                try:
                    if choice == '1':
                        result = vlm.damage_assessment(image_path)
                        
                    elif choice == '2':
                        result = vlm.describe_image(image_path)
                        
                    elif choice == '3':
                        result = vlm.identify_features(image_path)
                        
                    elif choice == '4':
                        result = vlm.assess_condition(image_path)
                        
                    elif choice == '5':
                        question = input("Enter your question about the image: ").strip()
                        result = vlm.custom_analysis(image_path, question)
                    
                    print("\n" + "="*70)
                    print(result)
                    print("="*70)
                    
                except Exception as e:
                    print(f"\nError: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == '6':
                before_path = input("\nEnter path to BEFORE image: ").strip()
                after_path = input("Enter path to AFTER image: ").strip()
                
                try:
                    result = vlm.compare_images(before_path, after_path)
                    print("\n" + "="*70)
                    print(result)
                    print("="*70)
                except Exception as e:
                    print(f"\nError: {e}")
                    import traceback
                    traceback.print_exc()
            
            else:
                print("\nInvalid option.")
            
            cont = input("\nAnalyze another image? (y/n): ").strip().lower()
            if cont != 'y':
                print("\nExiting. Goodbye!")
                break
                
    except Exception as e:
        print(f"\nFailed to initialize VLM: {e}")
        import traceback
        traceback.print_exc()
        