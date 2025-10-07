# Complete main.py with Replicate API integration and prompt.json usage
print("Starting main.py with Replicate API...")

import sys
import os
import logging
import traceback
from datetime import datetime
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import cv2
import numpy as np
import io
import uuid
import mediapipe as mp
import asyncio
from PIL import Image, ImageDraw
import uvicorn
from scipy import ndimage
import replicate

# Set up clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hair_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN environment variable not set")
    
# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
logger.info("Replicate API configured successfully")
MODEL_NAME = "black-forest-labs/flux-kontext-dev"
    
logger.info("All imports successful")

# Create FastAPI app
app = FastAPI(title="Hair Growth Simulation API with Replicate")
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
    
# Configuration
GENERATION_CONFIG = {
    "quality_check_delay": 2,
    "generation_timeout": 120
}

def load_prompts_config():
    """Load prompts configuration"""
    try:
        with open("prompt.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info("PROMPTS: Configuration loaded successfully")
            return config
    except FileNotFoundError:
        logger.warning("WARNING: prompt.json not found - using fallback prompts")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"ERROR: Invalid JSON in prompt.json: {e}")
        return {}

# Global variable to store prompts config
PROMPTS_CONFIG = load_prompts_config()
    
logger.info("PROMPTS: Configuration loaded" if PROMPTS_CONFIG else "PROMPTS: Using default configuration")
    
# Create necessary directories
directories = ["image_logs", "uploads", "detections", "logs/saved_settings"]
for directory in directories:
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
    
logger.info("Initialization completed successfully")

# ============= REPLICATE API FUNCTIONS =============
def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

def upload_image_to_data_url(image_data: bytes) -> str:
    """Convert image bytes to base64 data URL for Replicate"""
    img = Image.open(io.BytesIO(image_data))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

async def generate_with_replicate(input_image_data: bytes, mask_image_data: bytes, 
                                  prompt: str, request_id: str = "") -> bytes:
    """Core Replicate generation function using Flux Kontext Dev"""
    logger.info(f"REPLICATE-{request_id}: Starting generation with Flux Kontext Dev")

    try:
        # Convert input image to data URL (mask is not used with Flux model)
        input_url = upload_image_to_data_url(input_image_data)

        logger.info(f"REPLICATE-{request_id}: Sending to API...")
        logger.info(f"  - Prompt length: {len(prompt)} characters")

        # Log the complete prompt for debugging
        logger.info(f"REPLICATE-{request_id}: === PROMPT BEING SENT TO REPLICATE ===")
        logger.info(f"PROMPT: {prompt}")
        logger.info(f"REPLICATE-{request_id}: === END PROMPT ===")

        # Run the Flux model with correct input format
        output = replicate.run(
            MODEL_NAME,
            input={
                "prompt": prompt,
                "input_image": input_url,
                "output_format": "jpg",
                "num_inference_steps": 30
            }
        )

        # Handle Flux model return format - returns a URL directly
        if isinstance(output, str) and output.startswith("http"):
            # Output is a URL, download the image
            logger.info(f"REPLICATE-{request_id}: Downloading from URL")
            import urllib.request
            with urllib.request.urlopen(output) as response:
                generated_image_data = response.read()
        elif hasattr(output, 'read'):
            # If output is a file-like object, read it directly
            logger.info(f"REPLICATE-{request_id}: Reading from file object")
            generated_image_data = output.read()
        else:
            logger.error(f"REPLICATE-{request_id}: Unexpected output format: {type(output)}")
            return None

        if not generated_image_data or len(generated_image_data) < 5000:
            logger.error(f"REPLICATE-{request_id}: Generated image too small or empty")
            return None

        logger.info(f"REPLICATE-{request_id}: ✓ SUCCESS ({len(generated_image_data)/1024:.1f}KB)")
        return generated_image_data

    except Exception as e:
        logger.error(f"REPLICATE-{request_id}: Generation failed: {str(e)}")
        return None

def create_mask_from_coordinates(coordinates_data: dict, image_width: int, image_height: int, request_id: str = "") -> bytes:
    """Convert coordinate regions to a mask image for Replicate"""
    logger.info(f"MASK-CREATE-{request_id}: Creating mask from coordinates")
    
    try:
        regions = coordinates_data.get("regions", [])
        if not regions:
            logger.error(f"MASK-CREATE-{request_id}: No regions provided")
            return None
        
        # Create black background (preserve original)
        mask = Image.new('RGB', (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        # Draw white regions where hair should be generated
        for region in regions:
            x, y, w, h = region["bbox"]
            draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 255))
        
        # Convert to bytes
        buffered = io.BytesIO()
        mask.save(buffered, format="PNG")
        mask_data = buffered.getvalue()
        
        logger.info(f"MASK-CREATE-{request_id}: Created mask ({len(mask_data)/1024:.1f}KB)")
        return mask_data
        
    except Exception as e:
        logger.error(f"MASK-CREATE-{request_id}: Error creating mask: {str(e)}")
        return None

# ============= PROMPT BUILDING FUNCTIONS WITH PROMPT.JSON =============

def format_coordinates_for_prompt(coordinates_data: dict, request_id: str = "") -> str:
    """Generate coordinate prompt with priority-based density multipliers"""
    if not coordinates_data or "regions" not in coordinates_data:
        return "[NO MARKINGS] Use general hair growth approach."
    
    regions = coordinates_data["regions"]
    if not regions:
        return "[NO REGIONS] Use general hair growth approach."
    
    logger.info(f"PROMPT-{request_id}: Formatting {len(regions)} enhanced regions")
    
    priority_counts = coordinates_data.get("priority_distribution", {})
    total_area = coordinates_data.get("total_marked_area", 0)
    coverage_ratio = coordinates_data.get("coverage_ratio", 0)
    
    prompt_parts = [
        f"[ENHANCED COORDINATE-BASED HAIR GENERATION]",
        f"Generate hair in EXACTLY {len(regions)} marked regions:",
        f"Total marked area: {total_area} pixels ({coverage_ratio:.1%} coverage)",
        ""
    ]
    
    for i, region in enumerate(regions, 1):
        x, y, w, h = region["bbox"]
        center_x, center_y = region["center"]
        priority = region.get("density_priority", "medium")
        multiplier = region.get("density_multiplier", 1.0)
        
        prompt_parts.extend([
            f"REGION {i} - {priority.upper()} PRIORITY:",
            f"  • Boundaries: X={x} to {x+w}, Y={y} to {y+h}",
            f"  • Density multiplier: {multiplier}x",
            ""
        ])
    
    return "\n".join(prompt_parts)

def build_freemark_generation_prompt(coordinates_data: dict, timeframe: str,
                                   hair_density: float, hair_type: str,
                                   hair_color: str, request_id: str = "") -> str:
    """Build FreeMark prompt using hardcoded prompts based on timeframe"""
    logger.info(f"PROMPT-{request_id}: Building FreeMark prompt for {timeframe}")

    try:
        # Format coordinate prompt
        coord_prompt = format_coordinates_for_prompt(coordinates_data, request_id)

        # Use hardcoded prompts based on timeframe
        if timeframe in ["3months", "3 months"]:
            logger.info(f"PROMPT-{request_id}: Using 3-MONTH PROMPT")
            prompt = f"""Generate **subtle, natural hair growth** in the area colored #BCF473 (masked area) as it would appear after 3 months. Show **very short, fine, barely emerging hairs**, only within the masked region, following the exact direction, texture, color, thickness, and shine of the surrounding hair. **Completely remove the green mask color and outline**, replacing it seamlessly with natural hair—no traces of the mask should remain. **Do not generate hair outside the masked area**. **Keep all other regions completely unchanged**, preserving color, lighting, and composition. Ensure the new hairs are **extremely subtle and just beginning to grow**, reflecting minimal 3-month growth, fully natural and integrated.

{coord_prompt}

Hair Specifications:
- Color: {hair_color}
- Texture: {hair_type}
- Density: {hair_density * 100:.1f}%"""
        else:  # 8months
            logger.info(f"PROMPT-{request_id}: Using 8-MONTH PROMPT")
            prompt = f"""Generate **fully visible, natural hair growth** in the area colored #BCF473 (masked area) as it would appear after 8 months. Show **longer, thicker, and dense hairs**, seamlessly integrated with the surrounding hair, following the exact direction, texture, color, thickness, and shine. **Completely remove the green mask color and outline**, replacing it naturally with fully grown hair—no traces of the mask should remain. **Do not generate hair outside the masked area**. **Keep all other regions completely unchanged**, preserving color, lighting, and composition. Ensure the new hair looks **healthy, fully grown, clearly visible, and perfectly natural**, blending smoothly with existing hair.

{coord_prompt}

Hair Specifications:
- Color: {hair_color}
- Texture: {hair_type}
- Density: {hair_density * 100:.1f}%"""

        logger.info(f"PROMPT-{request_id}: Generated FreeMark prompt using hardcoded config ({len(prompt)} chars)")
        return prompt

    except Exception as e:
        logger.error(f"PROMPT-{request_id}: Error using hardcoded config: {str(e)}")
        return build_freemark_prompt_fallback(coordinates_data, timeframe, hair_density,
                                             hair_type, hair_color, request_id)

def build_freemark_prompt_fallback(coordinates_data: dict, timeframe: str, 
                                  hair_density: float, hair_type: str, 
                                  hair_color: str, request_id: str = "") -> str:
    """Fallback FreeMark prompt if config not available"""
    coord_prompt = format_coordinates_for_prompt(coordinates_data, request_id)
    
    if timeframe in ["3months", "3 months"]:
        base_multiplier = 1.25
        maturity_desc = "EARLY GROWTH STAGE"
    else:
        base_multiplier = 1.5
        maturity_desc = "MATURE GROWTH STAGE"
    
    effective_density = hair_density * base_multiplier
    
    return f"""PRECISE HAIR RESTORATION - {maturity_desc}

{coord_prompt}

SPECIFICATIONS:
✅ DENSITY: {hair_density * 100:.1f}% base coverage
✅ COLOR: {hair_color}
✅ TEXTURE: {hair_type}
✅ TIMEFRAME: {timeframe} progression

Generate natural {timeframe} hair growth matching existing style."""

def build_mask_based_hairline_prompt(timeframe: str, hair_density: float, hair_type: str,
                                   hair_color: str, pattern_type: str, request_id: str = "",
                                   input_image_size: int = 0, mask_image_size: int = 0,
                                   white_pixel_count: int = 0, total_pixels: int = 0) -> str:
    """Build prompt for mask-based generation using hardcoded prompts"""
    logger.info(f"MASK-PROMPT-{request_id}: Building prompt for {pattern_type} - {timeframe}")

    try:
        # Validate density
        hair_density = max(0.1, min(1.0, hair_density))

        # Use hardcoded prompts based on timeframe
        if timeframe in ["3months", "3 months"]:
            logger.info(f"MASK-PROMPT-{request_id}: Using 3-MONTH PROMPT for {pattern_type}")
            prompt = f"""Generate **subtle, natural hair growth** in the area colored #BCF473 (masked area) as it would appear after 3 months. Show **very short, fine, barely emerging hairs**, only within the masked region, following the exact direction, texture, color, thickness, and shine of the surrounding hair. **Completely remove the green mask color and outline**, replacing it seamlessly with natural hair—no traces of the mask should remain. **Do not generate hair outside the masked area**. **Keep all other regions completely unchanged**, preserving color, lighting, and composition. Ensure the new hairs are **extremely subtle and just beginning to grow**, reflecting minimal 3-month growth, fully natural and integrated.

Pattern: {pattern_type}
Hair Specifications:
- Color: {hair_color}
- Texture: {hair_type}
- Density: {hair_density * 100:.1f}%"""
        else:  # 8months
            logger.info(f"MASK-PROMPT-{request_id}: Using 8-MONTH PROMPT for {pattern_type}")
            prompt = f"""Generate **fully visible, natural hair growth** in the area colored #BCF473 (masked area) as it would appear after 8 months. Show **longer, thicker, and dense hairs**, seamlessly integrated with the surrounding hair, following the exact direction, texture, color, thickness, and shine. **Completely remove the green mask color and outline**, replacing it naturally with fully grown hair—no traces of the mask should remain. **Do not generate hair outside the masked area**. **Keep all other regions completely unchanged**, preserving color, lighting, and composition. Ensure the new hair looks **healthy, fully grown, clearly visible, and perfectly natural**, blending smoothly with existing hair.

Pattern: {pattern_type}
Hair Specifications:
- Color: {hair_color}
- Texture: {hair_type}
- Density: {hair_density * 100:.1f}%"""

        logger.info(f"MASK-PROMPT-{request_id}: Generated prompt using hardcoded config ({len(prompt)} chars)")
        return prompt

    except Exception as e:
        logger.error(f"MASK-PROMPT-{request_id}: Error using hardcoded config: {str(e)}")
        return build_mask_prompt_fallback(timeframe, hair_density, hair_type,
                                         hair_color, pattern_type, request_id)

def build_mask_prompt_fallback(timeframe: str, hair_density: float, hair_type: str, 
                               hair_color: str, pattern_type: str, request_id: str = "") -> str:
    """Fallback mask prompt if config not available"""
    if timeframe in ["3months", "3 months"]:
        enhancement_multiplier = 1.25
        maturity_desc = "EARLY GROWTH STAGE"
    else:
        enhancement_multiplier = 1.5
        maturity_desc = "MATURE GROWTH STAGE"
    
    effective_density = hair_density * enhancement_multiplier
    effective_percentage = min(100, effective_density * 100)
    
    return f"""PROFESSIONAL HAIR RESTORATION - {maturity_desc} FOR {pattern_type.upper()}

MISSION: Generate natural {timeframe} {pattern_type.lower()} hair restoration

MASK-BASED GENERATION:
- Image 1: Patient photo
- Image 2: Mask with BLACK background and WHITE areas = hair generation zones
- Generate hair ONLY in WHITE mask areas

PARAMETERS:
• HAIR COLOR: {hair_color}
• HAIR TEXTURE: {hair_type}
• HAIR DENSITY: {effective_percentage:.0f}% coverage
• TIMEFRAME: {timeframe} {maturity_desc.lower()}

Generate natural {timeframe} {pattern_type} restoration."""

# ============= GENERATION FUNCTIONS =============

async def generate_freemark_hair(input_image_data: bytes, input_mime_type: str,
                                coordinates_data: dict, timeframe: str, 
                                hair_density: float, hair_type: str, 
                                hair_color: str, request_id: str = ""):
    """FreeMark generation using Replicate API"""
    logger.info(f"FREEMARK-{request_id}: Starting FreeMark generation with Replicate")
    
    try:
        # Validate inputs
        if hair_density <= 0 or hair_density > 1:
            hair_density = max(0.1, min(1.0, hair_density))
        
        # Get image dimensions
        img = Image.open(io.BytesIO(input_image_data))
        image_width, image_height = img.size
        
        # Build prompt using prompt.json config
        prompt = build_freemark_generation_prompt(
            coordinates_data, timeframe, hair_density, 
            hair_type, hair_color, request_id
        )
        
        # Create mask from coordinates
        mask_data = create_mask_from_coordinates(
            coordinates_data, image_width, image_height, request_id
        )
        
        if not mask_data:
            logger.error(f"FREEMARK-{request_id}: Failed to create mask")
            return None
        
        # Generate with Replicate
        generated_image = await generate_with_replicate(
            input_image_data, mask_data, prompt, request_id
        )
        
        return generated_image
        
    except Exception as e:
        logger.error(f"FREEMARK-{request_id}: Error: {str(e)}")
        return None

async def generate_hairline_with_mask_enhanced(input_image_data: bytes, input_mime_type: str,
                                              mask_image_data: bytes, timeframe: str, 
                                              hair_density: float, hair_type: str, 
                                              hair_color: str, pattern_type: str, 
                                              request_id: str = ""):
    """Mask-based generation using Replicate API"""
    logger.info(f"MASK-GEN-{request_id}: Starting mask-based generation with Replicate")
    
    try:
        # Validate inputs
        if hair_density <= 0 or hair_density > 1:
            hair_density = max(0.1, min(1.0, hair_density))
        
        if not mask_image_data or len(mask_image_data) < 100:
            logger.error(f"MASK-GEN-{request_id}: Invalid mask data")
            return None
        
        # Build prompt using prompt.json config
        prompt = build_mask_based_hairline_prompt(
            timeframe, hair_density, hair_type, 
            hair_color, pattern_type, request_id
        )
        
        # Generate with Replicate
        generated_image = await generate_with_replicate(
            input_image_data, mask_image_data, prompt, request_id
        )
        
        return generated_image
        
    except Exception as e:
        logger.error(f"MASK-GEN-{request_id}: Error: {str(e)}")
        return None

# ============= FACE DETECTION FUNCTIONS =============

def detect_face_with_mediapipe(image_data: bytes, request_id: str = "") -> dict:
    """Detect face using MediaPipe"""
    logger.info(f"FACE-DETECT-{request_id}: Starting face detection...")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "face_detected": False,
                "confidence": 0.0,
                "error": "Failed to decode image"
            }

        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {
                    "face_detected": False,
                    "confidence": 0.0,
                    "error": "No face detected"
                }
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            face_left = min(xs) * w
            face_right = max(xs) * w
            face_top = min(ys) * h
            face_bottom = max(ys) * h
            
            face_width = face_right - face_left
            face_height = face_bottom - face_top
            center_x = int((face_left + face_right) / 2)
            center_y = int((face_top + face_bottom) / 2)
            
            face_area_ratio = (face_width * face_height) / (w * h)
            confidence = min(0.95, max(0.6, face_area_ratio * 10))
            
            # Save visualization
            vis_image = image.copy()
            cv2.rectangle(vis_image, 
                         (int(face_left), int(face_top)), 
                         (int(face_right), int(face_bottom)), 
                         (0, 255, 0), 2)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection_filename = f"{timestamp}_{request_id}_face_detection.jpg"
            detection_path = os.path.join("detections", detection_filename)
            cv2.imwrite(detection_path, vis_image)
            
            logger.info(f"FACE-DETECT-{request_id}: ✓ Face detected ({confidence:.1%})")
            
            return {
                "face_detected": True,
                "confidence": float(confidence),
                "landmarks_count": int(len(landmarks)),
                "face_bounds": {
                    "left": float(face_left),
                    "right": float(face_right), 
                    "top": float(face_top),
                    "bottom": float(face_bottom),
                    "width": float(face_width),
                    "height": float(face_height),
                    "center_x": int(center_x),
                    "center_y": int(center_y)
                },
                "face_area_ratio": float(face_area_ratio),
                "image_dimensions": {"width": int(w), "height": int(h)},
                "detection_image": detection_path
            }
            
    except Exception as e:
        logger.error(f"FACE-DETECT-{request_id}: Error: {str(e)}")
        return {
            "face_detected": False,
            "confidence": 0.0,
            "error": f"Detection error: {str(e)}"
        }

def generate_hairlines_and_scalp_regions(image_data: bytes, face_bounds: dict, request_id: str = "") -> dict:
    """Generate hairline patterns with filled regions"""
    logger.info(f"HAIRLINE-GEN-{request_id}: Generating hairline patterns...")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Failed to decode image"}

        h, w, _ = image.shape
        
        face_left = face_bounds["left"]
        face_right = face_bounds["right"]
        face_top = face_bounds["top"]
        face_width = face_bounds["width"]
        face_height = face_bounds["height"]
        center_x = int(face_bounds["center_x"])
        
        generated_patterns = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # M Pattern
        img_m = image.copy()
        m_points = [
            (int(face_left + face_width * 0.1), int(face_top - 20)),
            (int(face_left + face_width * 0.25), int(face_top - 40)),
            (int(center_x), int(face_top - 10)),
            (int(face_right - face_width * 0.25), int(face_top - 40)),
            (int(face_right - face_width * 0.1), int(face_top - 20))
        ]
        cv2.polylines(img_m, [np.array(m_points, np.int32)], False, (0, 255, 0), 2)
        m_filename = f"{timestamp}_{request_id}_hairline_m.jpg"
        m_path = os.path.join("detections", m_filename)
        cv2.imwrite(m_path, img_m)
        generated_patterns["hairline_m"] = m_path

        # Crown
        img_crown = image.copy()
        crown_y = int(face_top - (face_width * 0.5))
        crown_center = (center_x, crown_y)
        axes_crown = (int(face_width * 0.35), int(face_width * 0.2))
        cv2.ellipse(img_crown, crown_center, axes_crown, 0, 0, 360, (0, 255, 0), -1)
        cv2.ellipse(img_crown, crown_center, axes_crown, 0, 0, 360, (0, 255, 0), 4)
        crown_filename = f"{timestamp}_{request_id}_crown.jpg"
        crown_path = os.path.join("detections", crown_filename)
        cv2.imwrite(crown_path, img_crown)
        generated_patterns["crown"] = crown_path

        # Full Scalp
        img_scalp = image.copy()
        scalp_center = (center_x, int(face_top - face_height * 0.6))
        axes_scalp = (int(face_width * 0.5), int(face_height * 0.7))
        cv2.ellipse(img_scalp, scalp_center, axes_scalp, 0, 0, 360, (0, 255, 0), -1)
        cv2.ellipse(img_scalp, scalp_center, axes_scalp, 0, 0, 360, (0, 255, 0), 4)
        scalp_filename = f"{timestamp}_{request_id}_scalp.jpg"
        scalp_path = os.path.join("detections", scalp_filename)
        cv2.imwrite(scalp_path, img_scalp)
        generated_patterns["full_scalp"] = scalp_path

        # Mid-Crown
        img_mid_crown = image.copy()
        mid_crown_center = (center_x, int(scalp_center[1] - axes_scalp[1] * 0.3))
        axes_mid = (int(face_width * 0.25), int(face_width * 0.12))
        cv2.ellipse(img_mid_crown, mid_crown_center, axes_mid, 0, 0, 360, (0, 255, 0), -1)
        cv2.ellipse(img_mid_crown, mid_crown_center, axes_mid, 0, 0, 360, (0, 255, 0), 4)
        mid_crown_filename = f"{timestamp}_{request_id}_mid_crown.jpg"
        mid_crown_path = os.path.join("detections", mid_crown_filename)
        cv2.imwrite(mid_crown_path, img_mid_crown)
        generated_patterns["mid_crown"] = mid_crown_path

        logger.info(f"HAIRLINE-GEN-{request_id}: Generated {len(generated_patterns)} patterns")

        return {
            "success": True,
            "patterns": generated_patterns,
            "pattern_count": len(generated_patterns)
        }

    except Exception as e:
        logger.error(f"HAIRLINE-GEN-{request_id}: Error: {str(e)}")
        return {"error": f"Pattern generation error: {str(e)}"}

def convert_to_freemark_style_mask(pattern_image_data: bytes, request_id: str = "", pattern_type: str = "M Pattern") -> bytes:
    """Convert pattern to FreeMark mask"""
    logger.info(f"FREEMARK-MASK-{request_id}: Converting {pattern_type} to mask")
    
    try:
        pattern_image = Image.open(io.BytesIO(pattern_image_data))
        width, height = pattern_image.size
        
        gray_pattern = pattern_image.convert('L')
        pattern_array = np.array(gray_pattern)
        
        freemark_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        if pattern_type.lower() in ['crown', 'full scalp', 'mid crown']:
            pattern_pixels = pattern_array > 30
            if np.any(pattern_pixels):
                coords = np.where(pattern_pixels)
                min_y, max_y = np.min(coords[0]), np.max(coords[0])
                min_x, max_x = np.min(coords[1]), np.max(coords[1])
                
                center_y = (min_y + max_y) // 2
                center_x = (min_x + max_x) // 2
                
                radius_y = (max_y - min_y) // 2 + 30
                radius_x = (max_x - min_x) // 2 + 30
                
                y_coords, x_coords = np.ogrid[:height, :width]
                ellipse_mask = ((x_coords - center_x) / radius_x)**2 + ((y_coords - center_y) / radius_y)**2 <= 1
                
                freemark_mask[ellipse_mask] = [255, 255, 255]
        else:
            threshold = 30
            pattern_pixels = pattern_array > threshold
            cleaned_pattern = ndimage.binary_opening(pattern_pixels, iterations=2)
            solid_pattern = ndimage.binary_dilation(cleaned_pattern, iterations=8)
            filled_pattern = ndimage.binary_fill_holes(solid_pattern)
            final_pattern = ndimage.binary_closing(filled_pattern, iterations=3)
            freemark_mask[final_pattern] = [255, 255, 255]
        
        final_mask = Image.fromarray(freemark_mask)
        mask_bytes = io.BytesIO()
        final_mask.save(mask_bytes, format='PNG')
        
        logger.info(f"FREEMARK-MASK-{request_id}: Converted {pattern_type} mask")
        return mask_bytes.getvalue()
        
    except Exception as e:
        logger.error(f"FREEMARK-MASK-{request_id}: Error: {str(e)}")
        return pattern_image_data

def generate_hairline_coordinates_from_face_detection(image_width: int, image_height: int, 
                                                     request_id: str = "", 
                                                     pattern_type: str = "M_pattern") -> dict:
    """Generate coordinates for hairline patterns"""
    logger.info(f"HAIRLINE-COORDS-{request_id}: Generating {pattern_type} coordinates")
    
    try:
        face_width = int(image_width * 0.4)
        face_height = int(image_height * 0.5)
        face_left = int(image_width * 0.3)
        face_top = int(image_height * 0.2)
        face_right = face_left + face_width
        center_x = face_left + face_width // 2
        
        if pattern_type == "M_pattern":
            hairline_points = [
                [face_left + face_width * 0.1, face_top - 20],
                [face_left + face_width * 0.25, face_top - 40], 
                [center_x, face_top - 10],
                [face_right - face_width * 0.25, face_top - 40],
                [face_right - face_width * 0.1, face_top - 20]
            ]
        elif pattern_type == "Z_pattern":
            hairline_points = [
                [face_left + face_width * 0.1, face_top - 35],
                [face_right - face_width * 0.1, face_top - 35],
                [face_left + face_width * 0.15, face_top - 10],
                [face_right - face_width * 0.1, face_top - 5]
            ]
        else:
            hairline_points = [
                [face_left + face_width * 0.1, face_top - 20],
                [face_left + face_width * 0.3, face_top - 15],
                [center_x, face_top - 10],
                [face_right - face_width * 0.3, face_top - 15],
                [face_right - face_width * 0.1, face_top - 20]
            ]
        
        regions = []
        total_pixels = image_width * image_height
        
        for i, point in enumerate(hairline_points):
            x, y = point
            region_size = 35 if pattern_type == "Z_pattern" else 30
            
            region_width = min(region_size * 2, image_width - max(0, x - region_size))
            region_height = min(region_size * 2, image_height - max(0, y - region_size))
            actual_area = region_width * region_height
            area_ratio = actual_area / total_pixels
            
            if area_ratio >= 0.05:
                priority = "ultra_high"
                density_multiplier = 1.8
            elif area_ratio >= 0.02:
                priority = "high" 
                density_multiplier = 1.5
            elif area_ratio >= 0.01:
                priority = "medium"
                density_multiplier = 1.2
            else:
                priority = "standard"
                density_multiplier = 1.0
            
            region = {
                "bbox": [max(0, x - region_size), max(0, y - region_size), 
                        min(region_size * 2, image_width - max(0, x - region_size)),
                        min(region_size * 2, image_height - max(0, y - region_size))],
                "center": [int(x), int(y)],
                "coordinates": [
                    [max(0, x - region_size), max(0, y - region_size)],
                    [min(image_width, x + region_size), max(0, y - region_size)],
                    [min(image_width, x + region_size), min(image_height, y + region_size)],
                    [max(0, x - region_size), min(image_height, y + region_size)]
                ],
                "area": int(actual_area),
                "area_ratio": area_ratio,
                "density_priority": priority,
                "density_multiplier": density_multiplier,
                "growth_direction": "forward_natural",
                "region_id": f"hairline_point_{i+1}"
            }
            regions.append(region)
        
        priority_distribution = {
            priority: len([r for r in regions if r["density_priority"] == priority])
            for priority in ["ultra_high", "high", "medium", "standard"]
        }
        
        logger.info(f"  - Generated {len(regions)} {pattern_type} regions")
        
        return {
            "regions": regions,
            "total_regions": len(regions),
            "generation_method": "hairline_face_detection",
            "hairline_pattern": pattern_type,
            "hairline_points": hairline_points,
            "total_marked_area": sum(r["area"] for r in regions),
            "coverage_ratio": sum(r["area"] for r in regions) / total_pixels,
            "priority_distribution": priority_distribution
        }
        
    except Exception as e:
        logger.error(f"HAIRLINE-COORDS-{request_id}: Error: {str(e)}")
        return {"error": f"Failed to generate coordinates: {str(e)}"}

def extract_mask_from_colored_image(image_data: bytes, mask_color_hex: str = '#BCF473', request_id: str = '') -> tuple[bytes, bytes]:
    """Extract mask and original image from single image with colored mask overlay"""
    logger.info(f"MASK-EXTRACT-{request_id}: Extracting mask from colored image")
    
    try:
        # Convert hex color to RGB
        mask_color_hex = mask_color_hex.lstrip('#')
        mask_r = int(mask_color_hex[0:2], 16)
        mask_g = int(mask_color_hex[2:4], 16)
        mask_b = int(mask_color_hex[4:6], 16)
        
        logger.info(f"MASK-EXTRACT-{request_id}: Target mask color RGB({mask_r}, {mask_g}, {mask_b})")
        
        # Load image
        original_image = Image.open(io.BytesIO(image_data))
        image_array = np.array(original_image)
        
        height, width, _ = image_array.shape
        
        # Create mask - find pixels that match the mask color exactly or within tolerance
        mask_array = np.zeros((height, width), dtype=np.uint8)
        
        # Find exact color matches (for precise mask detection)
        color_mask = (
            (image_array[:,:,0] == mask_r) & 
            (image_array[:,:,1] == mask_g) & 
            (image_array[:,:,2] == mask_b)
        )
        
        mask_array[color_mask] = 255
        
        # Add tolerance for color variations (in case of compression artifacts)
        tolerance = 3
        color_mask_tolerant = (
            (np.abs(image_array[:,:,0] - mask_r) <= tolerance) & 
            (np.abs(image_array[:,:,1] - mask_g) <= tolerance) & 
            (np.abs(image_array[:,:,2] - mask_b) <= tolerance)
        )
        mask_array[color_mask_tolerant] = 255
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
        
        # Create the mask image (black background, white mask areas)
        mask_image = Image.fromarray(mask_array, mode='L')
        mask_rgb = Image.new('RGB', (width, height), (0, 0, 0))
        mask_rgb.paste(mask_image, mask=mask_image)
        
        # Create original image with mask color removed - do this more carefully
        original_array = image_array.copy()
        
        # Get indices where mask exists (either exact or tolerant match)
        mask_indices = np.where(mask_array == 255)
        
        if len(mask_indices[0]) > 0:  # Only if we found mask pixels
            # Replace mask color pixels with white
            original_array[mask_indices[0], mask_indices[1]] = [255, 255, 255]
        
        # Convert back to PIL Images
        original_clean = Image.fromarray(original_array)
        
        # Save to bytes
        original_buffer = io.BytesIO()
        original_clean.save(original_buffer, format='PNG')
        original_bytes = original_buffer.getvalue()
        
        mask_buffer = io.BytesIO()
        mask_rgb.save(mask_buffer, format='PNG')
        mask_bytes = mask_buffer.getvalue()
        
        logger.info(f"MASK-EXTRACT-{request_id}: Extracted original ({len(original_bytes)} bytes) and mask ({len(mask_bytes)} bytes)")
        
        # Log mask statistics
        white_pixels = np.sum(mask_array == 255)
        total_pixels = width * height
        coverage = (white_pixels / total_pixels) * 100
        logger.info(f"MASK-EXTRACT-{request_id}: Mask covers {white_pixels} pixels ({coverage:.1f}% of image)")
        
        return original_bytes, mask_bytes
        
    except Exception as e:
        logger.error(f"MASK-EXTRACT-{request_id}: Error extracting mask: {str(e)}")
        # Return original image as-is if extraction fails
        return image_data, None

def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

def analyze_mask_white_areas(mask_data: bytes, request_id: str = "") -> str:
    """Analyze mask to find coordinates of white areas"""
    try:
        mask_image = Image.open(io.BytesIO(mask_data))
        mask_array = np.array(mask_image.convert('L'))

        # Find white pixels (255)
        white_pixels = np.where(mask_array == 255)

        if len(white_pixels[0]) == 0:
            return "No white areas found in mask"

        # Get bounding box of all white areas
        min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
        min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])

        # Calculate center and dimensions
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Count white pixels
        white_count = len(white_pixels[0])

        # Get image dimensions
        img_height, img_width = mask_array.shape

        info = (
            f"Mask size: {img_width}x{img_height}, "
            f"White pixels: {white_count}, "
            f"Bounding box: x={min_x}-{max_x}, y={min_y}-{max_y}, "
            f"Center: ({center_x}, {center_y}), "
            f"Dimensions: {width}x{height}"
        )

        logger.info(f"MASK-ANALYSIS-{request_id}: {info}")
        return info

    except Exception as e:
        logger.error(f"MASK-ANALYSIS-{request_id}: Error analyzing mask: {str(e)}")
        return f"Error analyzing mask: {str(e)}"

@app.post("/detect-face")
async def detect_face_endpoint(image: UploadFile = File(...)):
    """Detect face and generate patterns"""
    request_id = generate_request_id()
    logger.info(f"DETECT-{request_id}: NEW FACE DETECTION REQUEST")
    
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await image.read()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{request_id}_upload_{image.filename}"
        upload_path = os.path.join("uploads", upload_filename)
        
        with open(upload_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"INPUT-{request_id}: Saved ({len(image_data) / 1024:.1f} KB)")
        
        detection_result = detect_face_with_mediapipe(image_data, request_id)
        
        pattern_result = None
        if detection_result["face_detected"]:
            pattern_result = generate_hairlines_and_scalp_regions(
                image_data, detection_result["face_bounds"], request_id
            )
        
        return {
            "request_id": request_id,
            "filename": image.filename,
            "upload_path": upload_path,
            "face_detection": detection_result,
            "show_freemark_only": not detection_result["face_detected"],
            "available_options": ["FreeMark"] if not detection_result["face_detected"] else [
                "Hairline", "Crown", "Mid Crown", "Full Scalp", "FreeMark"
            ],
            "patterns": pattern_result,
            "image_info": {
                "size_kb": f"{len(image_data) / 1024:.1f}",
                "content_type": image.content_type
            }
        }
        
    except Exception as e:
        logger.error(f"DETECT-{request_id}: ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

@app.post("/save-settings")
async def save_settings(
    image_3months: UploadFile = File(...),
    image_8months: UploadFile = File(...),
    hair_color: str = Form("#000000"),
    hair_type: str = Form("Straight Hair"),
    hair_line_type: str = Form("Hairline"),
    hair_density_3m: float = Form(0.7),
    hair_density_8m: float = Form(0.9),
    timeframe: str = Form("3months"),
    face_detected: str = Form("false"),
    hairline_pattern: str = Form("M_pattern"),
    hairline_points: Optional[str] = Form(None)
):
    """Save settings and two combined images with mask overlays"""
    request_id = generate_request_id()
    logger.info(f"SAVE-SETTINGS-{request_id}: Saving settings with two combined images")

    try:
        # Read both combined images (original + mask overlay for each timeframe)
        image_3m_data = await image_3months.read()
        image_8m_data = await image_8months.read()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        settings_dir = os.path.join("logs", "saved_settings")
        os.makedirs(settings_dir, exist_ok=True)

        # Save both combined images
        combined_3m_path = os.path.join(settings_dir, f"{request_id}_combined_3months.png")
        combined_8m_path = os.path.join(settings_dir, f"{request_id}_combined_8months.png")

        with open(combined_3m_path, "wb") as f:
            f.write(image_3m_data)
        with open(combined_8m_path, "wb") as f:
            f.write(image_8m_data)

        # Get image dimensions
        input_image_3m = Image.open(io.BytesIO(image_3m_data))
        input_image_8m = Image.open(io.BytesIO(image_8m_data))
        image_width, image_height = input_image_3m.size

        settings_data = {
            "request_id": request_id,
            "hair_color": hair_color,
            "hair_type": hair_type,
            "hair_line_type": hair_line_type,
            "hair_density_3m": hair_density_3m,
            "hair_density_8m": hair_density_8m,
            "timeframe": timeframe,
            "face_detected": face_detected == "true",
            "hairline_pattern": hairline_pattern,
            "image_dimensions": {"width": image_width, "height": image_height},
            "saved_at": datetime.now().isoformat(),
            "combined_images": {
                "image_3months": combined_3m_path,
                "image_8months": combined_8m_path
            }
        }

        if hairline_points:
            try:
                points_data = json.loads(hairline_points)
                settings_data["hairline_points"] = points_data
            except json.JSONDecodeError:
                pass

        # Extract and analyze masks from both combined images
        mask_info = {}

        for timeframe_key, image_data, timeframe_name in [
            ("3months", image_3m_data, "3months"),
            ("8months", image_8m_data, "8months")
        ]:
            try:
                original_bytes, mask_bytes = extract_mask_from_colored_image(
                    image_data, '#BCF473', f"{request_id}-{timeframe_key}"
                )

                if mask_bytes is not None:
                    # Save extracted mask
                    mask_path = os.path.join(settings_dir, f"{request_id}_mask_{timeframe_key}.png")
                    with open(mask_path, "wb") as f:
                        f.write(mask_bytes)

                    # Analyze mask
                    mask_analysis = analyze_mask_white_areas(mask_bytes, f"{request_id}-{timeframe_key}")
                    mask_info[timeframe_key] = {
                        "extracted": True,
                        "mask_path": mask_path,
                        "analysis": mask_analysis
                    }

                    logger.info(f"SAVE-SETTINGS-{request_id}: Successfully extracted {timeframe_key} mask")
                else:
                    logger.info(f"SAVE-SETTINGS-{request_id}: No mask overlay detected in {timeframe_key} combined image")
                    mask_info[timeframe_key] = {"extracted": False}

            except Exception as e:
                logger.warning(f"SAVE-SETTINGS-{request_id}: Could not extract {timeframe_key} mask: {str(e)}")
                mask_info[timeframe_key] = {"extracted": False, "error": str(e)}

        settings_data["mask_info"] = mask_info

        settings_path = os.path.join(settings_dir, f"{request_id}_settings.json")
        with open(settings_path, "w") as f:
            json.dump(settings_data, f, indent=2)

        logger.info(f"SAVE-SETTINGS-{request_id}: ✓ Settings and combined images saved")

        return {
            "success": True,
            "request_id": request_id,
            "message": "Settings and combined images saved successfully",
            "saved_files": {
                "combined_3months": combined_3m_path,
                "combined_8months": combined_8m_path,
                "settings": settings_path
            },
            "image_info": {
                "size_kb_3m": f"{len(image_3m_data) / 1024:.1f}",
                "size_kb_8m": f"{len(image_8m_data) / 1024:.1f}",
                "content_type_3m": image_3months.content_type,
                "content_type_8m": image_8months.content_type,
                "dimensions": f"{image_width}x{image_height}"
            },
            "mask_extraction": mask_info
        }

    except Exception as e:
        logger.error(f"SAVE-SETTINGS-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")

@app.get("/view-saved-image/{request_id}")
async def view_saved_image(request_id: str, timeframe: str = "3months"):
    """View the saved combined image with mask overlay for specific timeframe"""
    try:
        settings_dir = os.path.join("logs", "saved_settings")
        combined_image_path = os.path.join(settings_dir, f"{request_id}_combined_{timeframe}.png")

        if not os.path.exists(combined_image_path):
            raise HTTPException(status_code=404, detail=f"Combined image for {timeframe} not found")

        return FileResponse(
            combined_image_path,
            media_type="image/png",
            filename=f"{request_id}_combined_{timeframe}.png"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"VIEW-IMAGE-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve image: {str(e)}")

@app.get("/view-extracted-mask/{request_id}")
async def view_extracted_mask(request_id: str, timeframe: str = "3months"):
    """View the extracted mask from a saved combined image for specific timeframe"""
    try:
        settings_dir = os.path.join("logs", "saved_settings")
        mask_path = os.path.join(settings_dir, f"{request_id}_mask_{timeframe}.png")

        if not os.path.exists(mask_path):
            raise HTTPException(status_code=404, detail=f"Extracted mask for {timeframe} not found")

        return FileResponse(
            mask_path,
            media_type="image/png",
            filename=f"{request_id}_mask_{timeframe}.png"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"VIEW-MASK-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve mask: {str(e)}")

@app.post("/generate-with-saved-pattern")
async def generate_with_saved_pattern(
    image_3months: UploadFile = File(...),
    image_8months: UploadFile = File(...),
    hair_color: str = Form("#000000"),
    hair_type: str = Form("Straight Hair"),
    hair_line_type: str = Form("Hairline"),
    hair_density_3m: float = Form(0.7),
    hair_density_8m: float = Form(0.9),
    timeframe: str = Form("3months"),
    face_detected: str = Form("false"),
    use_saved_pattern: str = Form("true")
):
    """Generate hair using saved patterns with separate combined images for each timeframe"""
    request_id = generate_request_id()
    logger.info(f"GENERATE-{request_id}: Starting generation with saved pattern")

    try:
        # Read both combined images (original + mask overlay for each timeframe)
        image_3m_data = await image_3months.read()
        image_8m_data = await image_8months.read()

        # Determine which image to use based on the requested timeframe
        target_image_data = image_3m_data if timeframe == "3months" else image_8m_data
        target_timeframe = "3months" if timeframe == "3months" else "8months"

        input_image = Image.open(io.BytesIO(target_image_data))
        image_width, image_height = input_image.size

        # Load saved settings
        settings_dir = os.path.join("logs", "saved_settings")
        if not os.path.exists(settings_dir):
            raise HTTPException(status_code=404, detail="No saved settings found")

        settings_files = [f for f in os.listdir(settings_dir) if f.endswith("_settings.json")]
        if not settings_files:
            raise HTTPException(status_code=404, detail="No saved settings found")

        latest_settings = max(
            settings_files,
            key=lambda f: os.path.getmtime(os.path.join(settings_dir, f))
        )
        settings_path = os.path.join(settings_dir, latest_settings)

        with open(settings_path, "r") as f:
            saved_settings = json.load(f)

        saved_hair_line_type = saved_settings.get("hair_line_type", "Hairline")

        # Extract original image and mask from the target combined image
        original_bytes, mask_bytes = extract_mask_from_colored_image(
            target_image_data, '#BCF473', f"{request_id}-{target_timeframe}"
        )

        if mask_bytes is None:
            raise HTTPException(status_code=400, detail=f"No mask overlay detected in {target_timeframe} image")

        logger.info(f"GENERATE-{request_id}: Successfully extracted mask from {target_timeframe} combined image")

        # Generate based on pattern type
        if saved_hair_line_type == "Hairline":
            saved_pattern = saved_settings.get("hairline_pattern", "M_pattern")

            generated_3m = await generate_hairline_with_mask_enhanced(
                original_bytes, image_3months.content_type, mask_bytes,
                "3months", hair_density_3m, hair_type, hair_color,
                saved_pattern, f"{request_id}-3m"
            )

            generated_8m = await generate_hairline_with_mask_enhanced(
                original_bytes, image_3months.content_type, mask_bytes,
                "8months", hair_density_8m, hair_type, hair_color,
                saved_pattern, f"{request_id}-8m"
            )

            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_3m if timeframe == "3months" else generated_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": f"combined_images_hairline_{saved_pattern}",
                "pattern_type": saved_pattern,
                "mask_info": {
                    "mask_color": "#BCF473",
                    "extraction_successful": True,
                    "timeframe_used": target_timeframe
                }
            }

        elif saved_hair_line_type in ["Crown", "Mid Crown", "Full Scalp"]:
            generated_3m = await generate_hairline_with_mask_enhanced(
                original_bytes, image_3months.content_type, mask_bytes,
                "3months", hair_density_3m, hair_type, hair_color,
                saved_hair_line_type, f"{request_id}-3m"
            )

            generated_8m = await generate_hairline_with_mask_enhanced(
                original_bytes, image_3months.content_type, mask_bytes,
                "8months", hair_density_8m, hair_type, hair_color,
                saved_hair_line_type, f"{request_id}-8m"
            )

            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_3m if timeframe == "3months" else generated_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": f"combined_images_{saved_hair_line_type.lower().replace(' ', '_')}",
                "pattern_type": saved_hair_line_type,
                "mask_info": {
                    "mask_color": "#BCF473",
                    "extraction_successful": True,
                    "timeframe_used": target_timeframe
                }
            }

        else:  # FreeMark mode
            generated_3m = await generate_hairline_with_mask_enhanced(
                original_bytes, image_3months.content_type, mask_bytes,
                "3months", hair_density_3m, hair_type, hair_color,
                "FreeMark", f"{request_id}-3m"
            )

            generated_8m = await generate_hairline_with_mask_enhanced(
                original_bytes, image_3months.content_type, mask_bytes,
                "8months", hair_density_8m, hair_type, hair_color,
                "FreeMark", f"{request_id}-8m"
            )

            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_3m if timeframe == "3months" else generated_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": "combined_images_freemark",
                "pattern_type": "FreeMark",
                "mask_info": {
                    "mask_color": "#BCF473",
                    "extraction_successful": True,
                    "timeframe_used": target_timeframe
                }
            }

    except Exception as e:
        logger.error(f"GENERATE-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate-single-image-with-mask")
async def generate_single_image_with_mask(
    image: UploadFile = File(...),
    hair_color: str = Form("#000000"),
    hair_type: str = Form("Straight Hair"),
    hair_density_3m: float = Form(0.7),
    hair_density_8m: float = Form(0.9),
    timeframe: str = Form("3months"),
    pattern_type: str = Form("FreeMark")
):
    """Generate hair using single image with embedded mask (#BCF473 color)"""
    request_id = generate_request_id()
    logger.info(f"GENERATE-SINGLE-{request_id}: Starting single-image generation for {pattern_type}")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read the uploaded image data
        image_data = await image.read()

        # Extract original image and mask from the colored image
        original_bytes, mask_bytes = extract_mask_from_colored_image(
            image_data, '#BCF473', f"{request_id}-extract"
        )

        if mask_bytes is None:
            raise HTTPException(status_code=400, detail="No mask color (#BCF473) found in image")

        logger.info(f"GENERATE-SINGLE-{request_id}: Successfully extracted mask")

        # Generate hair growth for both timeframes
        generated_3m = await generate_hairline_with_mask_enhanced(
            original_bytes, image.content_type, mask_bytes,
            "3months", hair_density_3m, hair_type, hair_color,
            pattern_type, f"{request_id}-3m"
        )

        generated_8m = await generate_hairline_with_mask_enhanced(
            original_bytes, image.content_type, mask_bytes,
            "8months", hair_density_8m, hair_type, hair_color,
            pattern_type, f"{request_id}-8m"
        )

        if not generated_3m or not generated_8m:
            raise HTTPException(status_code=500, detail="Hair generation failed")

        return {
            "request_id": request_id,
            "image": base64.b64encode(generated_3m if timeframe == "3months" else generated_8m).decode("utf-8"),
            "image_3months": base64.b64encode(generated_3m).decode("utf-8"),
            "image_8months": base64.b64encode(generated_8m).decode("utf-8"),
            "has_both_timeframes": True,
            "generation_mode": f"single_image_{pattern_type.lower()}",
            "pattern_type": pattern_type,
            "mask_info": {
                "mask_color": "#BCF473",
                "extraction_successful": True
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GENERATE-SINGLE-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate-3months")
async def generate_3months(
    image: UploadFile = File(...),
    hair_color: str = Form("#000000"),
    hair_type: str = Form("Straight Hair"),
    hair_line_type: str = Form("Hairline"),
    hair_density_3m: float = Form(0.7),
    face_detected: str = Form("false"),
    use_saved_pattern: str = Form("true")
):
    """Generate hair growth for 3 months only"""
    request_id = generate_request_id()
    logger.info(f"GENERATE-3M-{request_id}: Starting 3-month generation")
    
    try:
        input_data = await image.read()
        input_image = Image.open(io.BytesIO(input_data))
        image_width, image_height = input_image.size
        
        # Load saved settings (same logic as before)
        settings_dir = os.path.join("logs", "saved_settings")
        if not os.path.exists(settings_dir):
            raise HTTPException(status_code=404, detail="No saved settings found")
        
        settings_files = [f for f in os.listdir(settings_dir) if f.endswith("_settings.json")]
        if not settings_files:
            raise HTTPException(status_code=404, detail="No saved settings found")
        
        latest_settings = max(settings_files, key=lambda f: os.path.getmtime(os.path.join(settings_dir, f)))
        settings_path = os.path.join(settings_dir, latest_settings)
        
        with open(settings_path, "r") as f:
            saved_settings = json.load(f)
        
        saved_hair_line_type = saved_settings.get("hair_line_type", "Hairline")
        mask_paths = saved_settings.get("mask_paths", {})
        
        # Generate 3-month result based on pattern type
        if saved_hair_line_type in ["Crown", "Mid Crown", "Full Scalp"]:
            if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
                with open(mask_paths["hairline_mask"], "rb") as f:
                    mask_data = f.read()

                logger.info(f"GENERATE-3M-{request_id}: Loaded Hairline mask ({len(mask_data)} bytes) for {saved_hair_line_type}")

                generated_image = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, mask_data,
                    "3months", hair_density_3m, hair_type, hair_color,
                    saved_hair_line_type, request_id
                )

                if not generated_image:
                    raise HTTPException(status_code=500, detail=f"Mask generation failed for {saved_hair_line_type}")

            else:
                # Fallback logic for pattern generation
                pattern_mapping = {"Crown": "crown", "Mid Crown": "mid_crown", "Full Scalp": "full_scalp"}
                pattern_key = pattern_mapping.get(saved_hair_line_type)
                if os.path.exists("detections"):
                    pattern_files = [f for f in os.listdir("detections") if pattern_key in f.lower()]
                    if pattern_files:
                        latest = max(pattern_files, key=lambda x: os.path.getmtime(os.path.join("detections", x)))
                        with open(os.path.join("detections", latest), "rb") as f:
                            pattern_data = f.read()
                        mask_data = convert_to_freemark_style_mask(pattern_data, request_id, saved_hair_line_type)

                        logger.info(f"GENERATE-3M-{request_id}: Generated mask from pattern ({len(mask_data)} bytes) for {saved_hair_line_type}")

                        generated_image = await generate_hairline_with_mask_enhanced(
                            input_data, image.content_type, mask_data,
                            "3months", hair_density_3m, hair_type, hair_color,
                            saved_hair_line_type, request_id
                        )

                        if not generated_image:
                            raise HTTPException(status_code=500, detail=f"Pattern-based generation failed for {saved_hair_line_type}")
                    else:
                        raise HTTPException(status_code=404, detail=f"No {saved_hair_line_type} pattern found")
                else:
                    raise HTTPException(status_code=404, detail="No detection files found")

        elif saved_hair_line_type == "FreeMark":
            # For FreeMark, expect a combined image (original + green mask overlay)
            logger.info(f"GENERATE-3M-{request_id}: Processing FreeMark combined image for 3 months")

            # Extract the green mask from the combined image
            mask_data = extract_green_mask_from_combined_image(input_data, request_id)

            if not mask_data:
                raise HTTPException(status_code=400, detail="No green mask found in FreeMark image - ensure image has green overlay")

            logger.info(f"GENERATE-3M-{request_id}: Extracted FreeMark mask ({len(mask_data)} bytes)")

            # Use 3-month prompt for subtle hair growth
            prompt = f"""Generate **subtle, natural hair growth** in the area colored #BCF473 (masked area) as it would appear after 3 months. Show **very short, fine, barely emerging hairs**, only within the masked region, following the exact direction, texture, color, thickness, and shine of the surrounding hair. **Completely remove the green mask color and outline**, replacing it seamlessly with natural hair—no traces of the mask should remain. **Do not generate hair outside the masked area**. **Keep all other regions completely unchanged**, preserving color, lighting, and composition. Ensure the new hairs are **extremely subtle and just beginning to grow**, reflecting minimal 3-month growth, fully natural and integrated."""

            generated_image = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_data,
                "3months", hair_density_3m, hair_type, hair_color,
                "FreeMark", request_id, custom_prompt=prompt
            )

            if not generated_image:
                raise HTTPException(status_code=500, detail="FreeMark 3-month generation failed")

        else:  # Hairline
            saved_pattern = saved_settings.get("hairline_pattern", "M_pattern")

            if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
                with open(mask_paths["hairline_mask"], "rb") as f:
                    mask_data = f.read()

                logger.info(f"GENERATE-3M-{request_id}: Loaded Hairline mask ({len(mask_data)} bytes)")

                generated_image = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, mask_data,
                    "3months", hair_density_3m, hair_type, hair_color,
                    saved_pattern, request_id
                )

                if not generated_image:
                    raise HTTPException(status_code=500, detail=f"Hairline {saved_pattern} generation failed")
            else:
                # For mask-based generation, we require a mask file
                raise HTTPException(status_code=404, detail=f"Hairline mask not found for pattern: {saved_pattern}")

        if not generated_image:
            raise HTTPException(status_code=500, detail="Generation returned empty result")
        
        return {
            "request_id": request_id,
            "image": base64.b64encode(generated_image).decode("utf-8"),
            "timeframe": "3months",
            "generation_mode": f"mask_based_{saved_hair_line_type.lower().replace(' ', '_')}",
            "pattern_type": saved_hair_line_type
        }
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"GENERATE-3M-{request_id}: Error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"3-month generation failed: {error_msg}")

@app.post("/generate-8months")
async def generate_8months(
    image: UploadFile = File(...),
    hair_color: str = Form("#000000"),
    hair_type: str = Form("Straight Hair"),
    hair_line_type: str = Form("Hairline"),
    hair_density_8m: float = Form(0.9),
    face_detected: str = Form("false"),
    use_saved_pattern: str = Form("true")
):
    """Generate hair growth for 8 months only"""
    request_id = generate_request_id()
    logger.info(f"GENERATE-8M-{request_id}: Starting 8-month generation")
    
    try:
        input_data = await image.read()
        input_image = Image.open(io.BytesIO(input_data))
        image_width, image_height = input_image.size
        
        # Load saved settings (same logic as before)
        settings_dir = os.path.join("logs", "saved_settings")
        if not os.path.exists(settings_dir):
            raise HTTPException(status_code=404, detail="No saved settings found")
        
        settings_files = [f for f in os.listdir(settings_dir) if f.endswith("_settings.json")]
        if not settings_files:
            raise HTTPException(status_code=404, detail="No saved settings found")
        
        latest_settings = max(settings_files, key=lambda f: os.path.getmtime(os.path.join(settings_dir, f)))
        settings_path = os.path.join(settings_dir, latest_settings)
        
        with open(settings_path, "r") as f:
            saved_settings = json.load(f)
        
        saved_hair_line_type = saved_settings.get("hair_line_type", "Hairline")
        mask_paths = saved_settings.get("mask_paths", {})
        
        # Generate 8-month result based on pattern type
        if saved_hair_line_type in ["Crown", "Mid Crown", "Full Scalp"]:
            if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
                with open(mask_paths["hairline_mask"], "rb") as f:
                    mask_data = f.read()

                logger.info(f"GENERATE-8M-{request_id}: Loaded Hairline mask ({len(mask_data)} bytes) for {saved_hair_line_type}")

                generated_image = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, mask_data,
                    "8months", hair_density_8m, hair_type, hair_color,
                    saved_hair_line_type, request_id
                )

                if not generated_image:
                    raise HTTPException(status_code=500, detail=f"Mask generation failed for {saved_hair_line_type}")

            else:
                # Fallback logic for pattern generation
                pattern_mapping = {"Crown": "crown", "Mid Crown": "mid_crown", "Full Scalp": "full_scalp"}
                pattern_key = pattern_mapping.get(saved_hair_line_type)
                if os.path.exists("detections"):
                    pattern_files = [f for f in os.listdir("detections") if pattern_key in f.lower()]
                    if pattern_files:
                        latest = max(pattern_files, key=lambda x: os.path.getmtime(os.path.join("detections", x)))
                        with open(os.path.join("detections", latest), "rb") as f:
                            pattern_data = f.read()
                        mask_data = convert_to_freemark_style_mask(pattern_data, request_id, saved_hair_line_type)

                        logger.info(f"GENERATE-8M-{request_id}: Generated mask from pattern ({len(mask_data)} bytes) for {saved_hair_line_type}")

                        generated_image = await generate_hairline_with_mask_enhanced(
                            input_data, image.content_type, mask_data,
                            "8months", hair_density_8m, hair_type, hair_color,
                            saved_hair_line_type, request_id
                        )

                        if not generated_image:
                            raise HTTPException(status_code=500, detail=f"Pattern-based generation failed for {saved_hair_line_type}")
                    else:
                        raise HTTPException(status_code=404, detail=f"No {saved_hair_line_type} pattern found")
                else:
                    raise HTTPException(status_code=404, detail="No detection files found")

        elif saved_hair_line_type == "FreeMark":
            # For FreeMark, expect a combined image (original + green mask overlay)
            logger.info(f"GENERATE-8M-{request_id}: Processing FreeMark combined image for 8 months")

            # Extract the green mask from the combined image
            mask_data = extract_green_mask_from_combined_image(input_data, request_id)

            if not mask_data:
                raise HTTPException(status_code=400, detail="No green mask found in FreeMark image - ensure image has green overlay")

            logger.info(f"GENERATE-8M-{request_id}: Extracted FreeMark mask ({len(mask_data)} bytes)")

            # Use 8-month prompt for fully grown hair
            prompt = f"""Generate **fully visible, natural hair growth** in the area colored #BCF473 (masked area) as it would appear after 8 months. Show **longer, thicker, and dense hairs**, seamlessly integrated with the surrounding hair, following the exact direction, texture, color, thickness, and shine. **Completely remove the green mask color and outline**, replacing it naturally with fully grown hair—no traces of the mask should remain. **Do not generate hair outside the masked area**. **Keep all other regions completely unchanged**, preserving color, lighting, and composition. Ensure the new hair looks **healthy, fully grown, clearly visible, and perfectly natural**, blending smoothly with existing hair."""

            generated_image = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_data,
                "8months", hair_density_8m, hair_type, hair_color,
                "FreeMark", request_id, custom_prompt=prompt
            )

            if not generated_image:
                raise HTTPException(status_code=500, detail="FreeMark 8-month generation failed")

        else:  # Hairline
            saved_pattern = saved_settings.get("hairline_pattern", "M_pattern")

            if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
                with open(mask_paths["hairline_mask"], "rb") as f:
                    mask_data = f.read()

                logger.info(f"GENERATE-8M-{request_id}: Loaded Hairline mask ({len(mask_data)} bytes)")

                generated_image = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, mask_data,
                    "8months", hair_density_8m, hair_type, hair_color,
                    saved_pattern, request_id
                )

                if not generated_image:
                    raise HTTPException(status_code=500, detail=f"Hairline {saved_pattern} generation failed")
            else:
                # For mask-based generation, we require a mask file
                raise HTTPException(status_code=404, detail=f"Hairline mask not found for pattern: {saved_pattern}")

        if not generated_image:
            raise HTTPException(status_code=500, detail="Generation returned empty result")
        
        return {
            "request_id": request_id,
            "image": base64.b64encode(generated_image).decode("utf-8"),
            "timeframe": "8months",
            "generation_mode": f"mask_based_{saved_hair_line_type.lower().replace(' ', '_')}",
            "pattern_type": saved_hair_line_type
        }
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error occurred"
        logger.error(f"GENERATE-8M-{request_id}: Error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"8-month generation failed: {error_msg}")

def extract_green_mask_from_combined_image(combined_image_data: bytes, request_id: str = "") -> bytes:
    """Extract green mask from combined image (original + green overlay)"""
    logger.info(f"MASK-EXTRACT-{request_id}: Extracting green mask from combined image")

    try:
        # Load the combined image
        combined_image = Image.open(io.BytesIO(combined_image_data))
        combined_array = np.array(combined_image)

        # Find green pixels (#BCF473 - RGB: 188, 244, 115)
        # Add tolerance for compression artifacts (±3)
        green_mask = (
            (combined_array[:,:,0] >= 185) & (combined_array[:,:,0] <= 191) &  # R: 185-191 (188±3)
            (combined_array[:,:,1] >= 241) & (combined_array[:,:,1] <= 247) &  # G: 241-247 (244±3)
            (combined_array[:,:,2] >= 112) & (combined_array[:,:,2] <= 118)    # B: 112-118 (115±3)
        )

        if not np.any(green_mask):
            logger.warning(f"MASK-EXTRACT-{request_id}: No green mask pixels found")
            return None

        # Create white mask from green areas
        white_mask = np.zeros_like(combined_array[:,:,0], dtype=np.uint8)
        white_mask[green_mask] = 255

        # Create RGB mask image (white on black background)
        mask_rgb = np.zeros_like(combined_array)
        mask_rgb[green_mask] = [255, 255, 255]  # White

        # Convert back to PIL Image
        mask_image = Image.fromarray(mask_rgb, mode='RGB')

        # Save as PNG bytes
        mask_bytes = io.BytesIO()
        mask_image.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)

        logger.info(f"MASK-EXTRACT-{request_id}: Extracted mask ({len(mask_bytes.getvalue())} bytes)")
        return mask_bytes.getvalue()

    except Exception as e:
        logger.error(f"MASK-EXTRACT-{request_id}: Error extracting mask: {str(e)}")
        return None

# ============= MAIN =============

if __name__ == "__main__":
    logger.info("STARTUP: Hair Growth API with Single Image Mask Support")
    uvicorn.run(app, host="0.0.0.0", port=8000)