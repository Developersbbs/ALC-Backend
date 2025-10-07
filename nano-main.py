# Updated main.py with prompt.json integration for Gemini
print("Starting main.py with prompt.json integration...")

import sys
import os
import logging
import traceback
from datetime import datetime
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from typing import Optional
import cv2
import numpy as np
import io
import uuid
import mediapipe as mp
import asyncio
from PIL import Image
import uvicorn
from scipy import ndimage

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

try:
    GEMINI_API_KEY = "AIzaSyDVgVz33Ciqctk2pBaSLWsMdCiXvPU_7gw"
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY is required. Please set it in main.py.")
        raise ValueError("GEMINI_API_KEY is required. Please set it in main.py.")

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        raise

    MODEL_NAME = "gemini-2.5-flash-image-preview"
    
    logger.info("All imports successful")

    app = FastAPI(title="Hair Growth Simulation API with Face Detection")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    GENERATION_CONFIG = {
        "max_retries": 3,
        "quality_check_delay": 2,
        "generation_timeout": 120
    }
    
    directories = ["uploads", "detections", "logs/saved_settings"]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
    
    logger.info("Initialization completed successfully")

except Exception as e:
    logger.error(f"Initialization error: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    print(f"CRITICAL ERROR: {e}")
    print("Please check the error log above and ensure all dependencies are installed.")
    sys.exit(1)

def load_prompts_config():
    """Load prompts configuration from prompt.json"""
    try:
        with open("prompt.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info("PROMPTS: Configuration loaded successfully from prompt.json")
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

def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

# ============= PROMPT BUILDING WITH PROMPT.JSON =============

def build_mask_based_freemark_prompt(timeframe: str, hair_density: float, hair_type: str, hair_color: str, request_id: str = ""):
    """Build FreeMark prompt using mask_based_generation config"""
    logger.info(f"FREEMARK-PROMPT-{request_id}: Building prompt for {timeframe}")

    try:
        # Use mask_based_generation config for FreeMark mask-based generation
        config = PROMPTS_CONFIG["generation_prompts"]["mask_based_generation"]

        # Get timeframe-specific settings
        timeframe_config = config["timeframe_specs"][timeframe]
        freemark_config = config["region_focus"]["FreeMark"]

        # Get mask interpretation
        mask_config = config["mask_interpretation"]
        white_areas = mask_config["white_areas"]
        black_areas = mask_config["black_areas"]

        # Build the prompt with explicit mask instructions
        prompt_parts = [
            f"You are {config['system_role']}",
            "",
            f"{config['base_instruction']}",
            "",
            f"MISSION: {timeframe_config['generation_text']} using mask guidance",
            "",
            f"IMPORTANT: This is FreeMark generation - follow user-marked areas precisely!",
            "",
            f"IMAGE ANALYSIS:",
            f"- Image 1: Patient photo requiring hair restoration",
            f"- Image 2: Mask with BLACK background and WHITE marked areas",
            "",
            f"MASK INTERPRETATION:",
            f"WHITE AREAS - {white_areas['description']}:",
            f"  • Density Level: {white_areas['density_percentage']}%",
        ]

        # Add white area characteristics
        for char in white_areas['characteristics']:
            prompt_parts.append(f"  • {char}")

        prompt_parts.extend([
            "",
            f"BLACK AREAS - {black_areas['description']}:",
            f"  • Density Level: {black_areas['density_percentage']}%",
        ])

        # Add black area characteristics
        for char in black_areas['characteristics']:
            prompt_parts.append(f"  • {char}")

        prompt_parts.extend([
            "",
            "SPECIFICATIONS:",
            f"• HAIR COLOR: {hair_color}",
            f"• HAIR TEXTURE: {hair_type}",
            f"• BASE DENSITY: {hair_density * 100:.0f}%",
            f"• EFFECTIVE DENSITY: {hair_density * 100:.0f}%",
            f"• {timeframe_config['visibility_focus']}",
            "",
            "GENERATION STEPS:",
        ])

        # Add generation steps from config
        for step in config["generation_steps"]:
            prompt_parts.append(f"{step}")

        prompt_parts.append("")
        prompt_parts.append("SUCCESS CRITERIA:")

        # Add success factors from config
        for factor in config["success_factors"]:
            prompt_parts.append(f"✓ {factor}")

        prompt = "\n".join(prompt_parts)

        logger.info(f"FREEMARK-PROMPT-{request_id}: Generated using mask config ({len(prompt)} chars)")
        return prompt

    except Exception as e:
        logger.error(f"FREEMARK-PROMPT-{request_id}: Error using config: {str(e)}")
        return build_freemark_prompt_fallback(timeframe, hair_density, hair_type, hair_color, request_id)

def build_freemark_prompt_fallback(timeframe: str, hair_density: float, hair_type: str, 
                                   hair_color: str, request_id: str = "") -> str:
    """Fallback prompt if config not available"""
    if timeframe in ["3months", "3 months"]:
        enhancement_multiplier = 1.8
        maturity_desc = "3-MONTH VISIBLE GROWTH"
    else:
        enhancement_multiplier = 1.5
        maturity_desc = "8-MONTH MATURE GROWTH"
    
    effective_density = hair_density * enhancement_multiplier
    effective_percentage = min(100, effective_density * 100)
    
    return f"""FREEMARK HAIR RESTORATION - {maturity_desc}

MISSION: Generate natural {timeframe} hair restoration

MASK-BASED GENERATION:
- Image 1: Patient photo
- Image 2: Mask with BLACK background and WHITE areas = hair generation zones

SPECIFICATIONS:
• Color: {hair_color}
• Texture: {hair_type}
• Density: {effective_percentage:.0f}% coverage
• Timeframe: {timeframe}

Generate visible {timeframe} FreeMark restoration."""

def build_mask_based_hairline_prompt(timeframe: str, hair_density: float, hair_type: str, 
                                   hair_color: str, pattern_type: str, request_id: str = "",
                                   input_image_size: int = 0, mask_image_size: int = 0, 
                                   white_pixel_count: int = 0, total_pixels: int = 0) -> str:
    """Build hairline prompt using prompt.json config"""
    logger.info(f"HAIRLINE-PROMPT-{request_id}: Building prompt for {pattern_type} - {timeframe}")
    
    try:
        hair_density = max(0.1, min(1.0, hair_density))
        
        # Get config from prompt.json
        config = PROMPTS_CONFIG.get("generation_prompts", {}).get("mask_based_generation", {})
        
        if not config:
            logger.warning(f"HAIRLINE-PROMPT-{request_id}: No config found, using fallback")
            return build_hairline_prompt_fallback(timeframe, hair_density, hair_type, 
                                                 hair_color, pattern_type, request_id)
        
        # Get timeframe specs
        timeframe_key = "3months" if timeframe in ["3months", "3 months"] else "8months"
        timeframe_config = config.get("timeframe_specs", {}).get(timeframe_key, {})
        
        enhancement_multiplier = timeframe_config.get("enhancement_multiplier", 1.25)
        maturity_desc = timeframe_config.get("maturity", "growth stage")
        visibility_focus = timeframe_config.get("visibility_focus", "natural visibility")
        generation_text = timeframe_config.get("generation_text", "Generate natural hair restoration")
        
        # Get region-specific config
        region_config = config.get("region_focus", {}).get(pattern_type, {})
        region_description = region_config.get("description", f"Focus on {pattern_type} restoration")
        region_enhancement = region_config.get("enhancement_multiplier", 1.5)
        specific_instructions = region_config.get("specific_instructions", "")
        
        # Get mask interpretation
        mask_config = config.get("mask_interpretation", {})
        white_areas = mask_config.get("white_areas", {})
        black_areas = mask_config.get("black_areas", {})
        
        # Calculate effective density
        effective_density = hair_density * enhancement_multiplier * region_enhancement
        effective_percentage = min(100, effective_density * 100)
        
        # Build prompt using config
        prompt_parts = [
            f"{config.get('system_role', 'Hair restoration specialist')}",
            f"",
            f"{config.get('base_instruction', 'PROFESSIONAL HAIR RESTORATION')}",
            f"",
            f"MISSION: {generation_text}",
            f"PATTERN TYPE: {pattern_type.upper()}",
            f"REGION: {region_description}",
            f"TIMEFRAME: {timeframe} - {maturity_desc}",
            f"",
            f"MASK INTERPRETATION:",
            f"Image 1: Patient photo requiring restoration",
            f"Image 2: Mask with BLACK and WHITE areas",
            f"",
            f"WHITE AREAS - {white_areas.get('description', 'Generate hair here')}:",
            f"  • Density Level: {white_areas.get('density_percentage', 100)}%",
        ]
        
        for char in white_areas.get('characteristics', []):
            prompt_parts.append(f"  • {char}")
        
        prompt_parts.extend([
            f"",
            f"BLACK AREAS - {black_areas.get('description', 'Preserve original')}:",
            f"  • Density Level: {black_areas.get('density_percentage', '15-20')}%",
        ])
        
        for char in black_areas.get('characteristics', []):
            prompt_parts.append(f"  • {char}")
        
        prompt_parts.extend([
            f"",
            f"SPECIFICATIONS:",
            f"• HAIR COLOR: {hair_color}",
            f"• HAIR TEXTURE: {hair_type}",
            f"• BASE DENSITY: {hair_density * 100:.0f}%",
            f"• EFFECTIVE DENSITY: {effective_percentage:.0f}%",
            f"• {visibility_focus}",
            f"",
            f"REGION-SPECIFIC INSTRUCTIONS:",
            f"{specific_instructions}",
            f"",
            f"GENERATION STEPS:",
        ])
        
        for step in config.get("generation_steps", []):
            prompt_parts.append(f"{step}")
        
        prompt_parts.append(f"")
        prompt_parts.append(f"SUCCESS CRITERIA:")
        
        for factor in config.get("success_factors", []):
            prompt_parts.append(f"✓ {factor}")
        
        prompt = "\n".join(prompt_parts)
        
        logger.info(f"HAIRLINE-PROMPT-{request_id}: Generated using config ({len(prompt)} chars)")
        return prompt
        
    except Exception as e:
        logger.error(f"HAIRLINE-PROMPT-{request_id}: Error using config: {str(e)}")
        return build_hairline_prompt_fallback(timeframe, hair_density, hair_type, 
                                             hair_color, pattern_type, request_id)

def build_hairline_prompt_fallback(timeframe: str, hair_density: float, hair_type: str, 
                                   hair_color: str, pattern_type: str, request_id: str = "") -> str:
    """Fallback hairline prompt if config not available"""
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

PARAMETERS:
• HAIR COLOR: {hair_color}
• HAIR TEXTURE: {hair_type}
• HAIR DENSITY: {effective_percentage:.0f}% coverage
• TIMEFRAME: {timeframe}

Generate natural {timeframe} {pattern_type} restoration."""

# ============= EXISTING FUNCTIONS (keeping as is) =============

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

async def generate_hairline_with_mask_enhanced(input_image_data: bytes, input_mime_type: str,
                                    mask_image_data: bytes, timeframe: str, 
                                    hair_density: float, hair_type: str, 
                                    hair_color: str, pattern_type: str, request_id: str = "", custom_prompt: str = None):
    """Generate using mask with prompt.json integration"""
    logger.info(f"MASK-GEN-{request_id}: Starting mask-based generation for {pattern_type}")
    
    if hair_density <= 0 or hair_density > 1:
        logger.warning(f"MASK-GEN-{request_id}: Invalid density {hair_density}, clamping")
        hair_density = max(0.1, min(1.0, hair_density))
    
    if not mask_image_data or len(mask_image_data) < 100:
        logger.error(f"MASK-GEN-{request_id}: Invalid mask data")
        return None
    
    logger.info(f"MASK-GEN-{request_id}: Mask data size: {len(mask_image_data)} bytes")
    
    try:
        # Use custom prompt if provided, otherwise use standard prompt logic
        if custom_prompt:
            prompt = custom_prompt
            logger.info(f"MASK-GEN-{request_id}: Using custom prompt for {timeframe}")
        else:
            # Use appropriate prompt with prompt.json
            if pattern_type == "FreeMark":
                prompt = build_mask_based_freemark_prompt(timeframe, hair_density, hair_type, hair_color, request_id)
            else:
                prompt = build_mask_based_hairline_prompt(timeframe, hair_density, hair_type, hair_color, pattern_type, request_id)
        
        # Enhance mask
        enhanced_mask_data = convert_pattern_to_white_mask(mask_image_data, request_id)
        
        if len(enhanced_mask_data) < 100:
            logger.warning(f"MASK-GEN-{request_id}: Mask conversion failed, using original")
            enhanced_mask_data = mask_image_data
        
        # Analyze mask for white areas
        white_areas_info = analyze_mask_white_areas(enhanced_mask_data, request_id)
        logger.info(f"MASK-ANALYSIS-{request_id}: White areas found: {white_areas_info}")
        
        image_b64 = base64.b64encode(input_image_data).decode('utf-8')
        mask_b64 = base64.b64encode(enhanced_mask_data).decode('utf-8')
        
        content = [
            {
                "inline_data": {
                    "mime_type": input_mime_type,
                    "data": image_b64
                }
            },
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": mask_b64
                }
            },
            prompt
        ]
        
        logger.info(f"MASK-GEN-{request_id}: FULL PROMPT BEING USED:")
        logger.info(prompt)
        
        models_to_try = ["gemini-2.5-flash-image-preview"]
        
        for model_name in models_to_try:
            max_retries = 2
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"MASK-GEN-{request_id}: {model_name} attempt {attempt + 1}/{max_retries}")
                    
                    model = genai.GenerativeModel(model_name)
                    
                    generation_config = {
                        "temperature": 0.4,
                        "top_p": 0.95,
                        "top_k": 50,
                        "max_output_tokens": 4096,
                    }
                    
                    response = await asyncio.wait_for(
                        asyncio.to_thread(model.generate_content, content, generation_config=generation_config),
                        timeout=90
                    )
                    
                    if not response:
                        logger.warning(f"MASK-GEN-{request_id}: Empty response")
                        continue
                    
                    image_data = extract_and_validate_image_data(response, request_id)
                    
                    if image_data:
                        logger.info(f"MASK-GEN-{request_id}: ✓ SUCCESS with {model_name}")
                        logger.info(f"  - Generated: {len(image_data)/1024:.1f}KB image")
                        return image_data
                    else:
                        logger.warning(f"MASK-GEN-{request_id}: No valid image extracted")
                        
                except asyncio.TimeoutError:
                    logger.error(f"MASK-GEN-{request_id}: Timeout on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"MASK-GEN-{request_id}: Attempt {attempt + 1} failed: {str(e)}")
                    if "quota" in str(e).lower() or "limit" in str(e).lower():
                        break
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(1)
    
        logger.error(f"MASK-GEN-{request_id}: All generation attempts failed")
        return None
        
    except Exception as e:
        logger.error(f"MASK-GEN-{request_id}: Critical error: {str(e)}")
        return None

# [Rest of the functions remain the same - detect_face_with_mediapipe, generate_hairlines_and_scalp_regions,
# convert_to_freemark_style_mask, convert_pattern_to_white_mask, extract_and_validate_image_data, etc.]
# I'll include the key ones but keep them unchanged:

def detect_face_with_mediapipe(image_data: bytes, request_id: str = "") -> dict:
    """Detect face using MediaPipe"""
    logger.info(f"FACE-DETECT-{request_id}: Starting face detection...")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"face_detected": False, "confidence": 0.0, "error": "Failed to decode image"}

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
                return {"face_detected": False, "confidence": 0.0, "error": "No face detected"}
            
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
            
            vis_image = image.copy()
            cv2.rectangle(vis_image, (int(face_left), int(face_top)), (int(face_right), int(face_bottom)), (0, 255, 0), 2)
            
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
        return {"face_detected": False, "confidence": 0.0, "error": f"Detection error: {str(e)}"}

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
        
        if pattern_type.lower() in ['crown', 'full scalp', 'mid crown', 'full_scalp', 'mid_crown']:
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

def convert_pattern_to_white_mask(pattern_image_data: bytes, request_id: str = "") -> bytes:
    """Convert pattern to white mask for generation"""
    logger.info(f"MASK-CONVERT-{request_id}: Converting to white mask")
    
    try:
        pattern_image = Image.open(io.BytesIO(pattern_image_data))
        gray_pattern = pattern_image.convert('L')
        pattern_array = np.array(gray_pattern)
        
        threshold = 50
        white_mask = np.where(pattern_array > threshold, 255, 0).astype(np.uint8)
        
        kernel = np.ones((3,3), np.uint8)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        white_mask = cv2.erode(white_mask, kernel, iterations=1)
        
        mask_image = Image.fromarray(white_mask, mode='L')
        mask_rgb = Image.new('RGB', mask_image.size, (0, 0, 0))
        mask_rgb.paste(mask_image, mask=mask_image)
        
        mask_array = np.array(mask_rgb)
        mask_array[white_mask == 255] = [255, 255, 255]
        
        final_mask = Image.fromarray(mask_array)
        mask_bytes = io.BytesIO()
        final_mask.save(mask_bytes, format='PNG')
        
        logger.info(f"MASK-CONVERT-{request_id}: Converted ({len(mask_bytes.getvalue())} bytes)")
        return mask_bytes.getvalue()
        
    except Exception as e:
        logger.error(f"MASK-CONVERT-{request_id}: Error: {str(e)}")
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
    """Decode and validate image data"""
    try:
        decoded_data = None
        
        if isinstance(data, str):
            try:
                decoded_data = base64.b64decode(data)
            except Exception as e:
                logger.error(f"DECODE-{context}: Base64 decode failed: {e}")
                return None
        elif isinstance(data, bytes):
            decoded_data = data
        else:
            logger.warning(f"DECODE-{context}: Unknown data type: {type(data)}")
            return None
        
        if not decoded_data or len(decoded_data) < 5000:
            logger.warning(f"DECODE-{context}: Image too small: {len(decoded_data) if decoded_data else 0} bytes")
            return None
        
        try:
            test_image = Image.open(io.BytesIO(decoded_data))
            test_image.verify()
            logger.info(f"DECODE-{context}: Valid image: {test_image.size}, {len(decoded_data)} bytes")
            return decoded_data
        except Exception as e:
            logger.error(f"DECODE-{context}: Image validation failed: {e}")
            return None
            
    except Exception as e:
        logger.error(f"DECODE-{context}: Decode error: {str(e)}")
        return None

def decode_and_validate_image(data, context: str = ""):
    """IMPROVED: Decode and validate image data with better error handling"""
    try:
        decoded_data = None
        
        if isinstance(data, str):
            try:
                decoded_data = base64.b64decode(data)
            except Exception as e:
                logger.error(f"DECODE-{context}: Base64 decode failed: {e}")
                return None
        elif isinstance(data, bytes):
            decoded_data = data
        else:
            logger.warning(f"DECODE-{context}: Unknown data type: {type(data)}")
            return None
        
        if not decoded_data or len(decoded_data) < 5000:
            logger.warning(f"DECODE-{context}: Image too small: {len(decoded_data) if decoded_data else 0} bytes")
            return None
        
        # Validate that it's actually an image
        try:
            test_image = Image.open(io.BytesIO(decoded_data))
            test_image.verify()
            logger.info(f"DECODE-{context}: Valid image: {test_image.size}, {len(decoded_data)} bytes")
            return decoded_data
        except Exception as e:
            logger.error(f"DECODE-{context}: Image validation failed: {e}")
            return None
            
    except Exception as e:
        logger.error(f"DECODE-{context}: Decode error: {str(e)}")
        return None

def extract_and_validate_image_data(response, request_id: str = ""):
    """Extract image from Gemini response"""
    logger.info(f"EXTRACT-{request_id}: Extracting image data")
    
    try:
        if hasattr(response, 'parts') and response.parts:
            for i, part in enumerate(response.parts):
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'data') and part.inline_data.data:
                        image_data = part.inline_data.data
                        decoded_data = decode_and_validate_image(image_data, f"{request_id}-parts-{i}")
                        if decoded_data:
                            return decoded_data
        
        if hasattr(response, 'candidates') and response.candidates:
            for j, candidate in enumerate(response.candidates):
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for k, part in enumerate(candidate.content.parts):
                            if hasattr(part, 'inline_data') and part.inline_data:
                                if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                    image_data = part.inline_data.data
                                    decoded_data = decode_and_validate_image(image_data, f"{request_id}-candidate-{j}-{k}")
                                    if decoded_data:
                                        return decoded_data
        
        logger.warning(f"EXTRACT-{request_id}: No valid image data found")
        return None
        
    except Exception as e:
        logger.error(f"EXTRACT-{request_id}: Error: {str(e)}")
        return None

# ============= API ENDPOINTS =============

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
    image: UploadFile = File(...),
    hair_color: str = Form("#000000"),
    hair_type: str = Form("Straight Hair"),
    hair_line_type: str = Form("Hairline"),
    hair_density_3m: float = Form(0.7),
    hair_density_8m: float = Form(0.9),
    timeframe: str = Form("3months"),
    face_detected: str = Form("false"),
    mask_3months: Optional[UploadFile] = File(None),
    mask_8months: Optional[UploadFile] = File(None),
    hairline_mask: Optional[UploadFile] = File(None),
    hairline_pattern: str = Form("M_pattern"),
    hairline_points: Optional[str] = Form(None)
):
    """Save settings and masks"""
    request_id = generate_request_id()
    logger.info(f"SAVE-SETTINGS-{request_id}: Saving user settings")
    
    try:
        image_data = await image.read()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{request_id}_upload_{image.filename}"
        upload_path = os.path.join("uploads", upload_filename)
        
        with open(upload_path, "wb") as f:
            f.write(image_data)
        
        input_image = Image.open(io.BytesIO(image_data))
        image_width, image_height = input_image.size
        
        settings_dir = os.path.join("logs", "saved_settings")
        os.makedirs(settings_dir, exist_ok=True)
        
        original_path = os.path.join(settings_dir, f"{request_id}_original.png")
        input_image.save(original_path)
        
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
            "saved_at": datetime.now().isoformat()
        }
        
        if hairline_points:
            try:
                points_data = json.loads(hairline_points)
                settings_data["hairline_points"] = points_data
            except json.JSONDecodeError:
                pass
        
        mask_paths = {}
        
        if mask_3months:
            mask_3m_data = await mask_3months.read()
            mask_3m_path = os.path.join(settings_dir, f"{request_id}_mask_3months.png")
            with open(mask_3m_path, "wb") as f:
                f.write(mask_3m_data)
            mask_paths["mask_3months"] = mask_3m_path
            logger.info(f"  - Saved 3-month mask")
            
        if mask_8months:
            mask_8m_data = await mask_8months.read()
            mask_8m_path = os.path.join(settings_dir, f"{request_id}_mask_8months.png")
            with open(mask_8m_path, "wb") as f:
                f.write(mask_8m_data)
            mask_paths["mask_8months"] = mask_8m_path
            logger.info(f"  - Saved 8-month mask")
        
        if hairline_mask:
            hairline_mask_data = await hairline_mask.read()
            freemark_style_mask = convert_to_freemark_style_mask(
                hairline_mask_data, request_id, hairline_pattern
            )
            hairline_mask_path = os.path.join(settings_dir, f"{request_id}_hairline_mask.png")
            with open(hairline_mask_path, "wb") as f:
                f.write(freemark_style_mask)
            mask_paths["hairline_mask"] = hairline_mask_path
            logger.info(f"  - Saved {hairline_pattern} mask")
        
        settings_data["mask_paths"] = mask_paths
        
        settings_path = os.path.join(settings_dir, f"{request_id}_settings.json")
        with open(settings_path, "w") as f:
            json.dump(settings_data, f, indent=2)
        
        logger.info(f"SAVE-SETTINGS-{request_id}: ✓ Settings saved")
        
        return {
            "success": True,
            "request_id": request_id,
            "message": "Settings saved - use generate-with-saved-pattern endpoint",
            "saved_files": {
                "original_image": original_path,
                "settings": settings_path,
                "masks": mask_paths
            }
        }
        
    except Exception as e:
        logger.error(f"SAVE-SETTINGS-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")
# Add these new endpoints after the existing /generate-with-saved-pattern endpoint

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
                    else:
                        raise HTTPException(status_code=404, detail=f"No {saved_hair_line_type} pattern found")
                else:
                    raise HTTPException(status_code=404, detail="No detection files found")
            
        elif saved_hair_line_type == "FreeMark":
            if "mask_3months" not in mask_paths or not os.path.exists(mask_paths["mask_3months"]):
                raise HTTPException(status_code=404, detail="FreeMark 3-month mask not found")
            
            with open(mask_paths["mask_3months"], "rb") as f:
                mask_data = f.read()
            
            logger.info(f"GENERATE-3M-{request_id}: Loaded FreeMark 3-month mask ({len(mask_data)} bytes)")
            
            generated_image = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_data,
                "3months", hair_density_3m, hair_type, hair_color, 
                "FreeMark", request_id
            )
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
            else:
                # Fallback to coordinate generation
                coords = generate_hairline_coordinates_from_face_detection(
                    image_width, image_height, request_id, saved_pattern
                )
                
                # Note: This would need the generate_freemark_hair function if you want to support it
                # For now, assuming mask-based generation
                raise HTTPException(status_code=404, detail="Hairline mask not found")
        
        if not generated_image:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        return {
            "request_id": request_id,
            "image": base64.b64encode(generated_image).decode("utf-8"),
            "timeframe": "3months",
            "generation_mode": f"mask_based_{saved_hair_line_type.lower().replace(' ', '_')}",
            "pattern_type": saved_hair_line_type
        }
        
    except Exception as e:
        logger.error(f"GENERATE-3M-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"3-month generation failed: {str(e)}")

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
                    else:
                        raise HTTPException(status_code=404, detail=f"No {saved_hair_line_type} pattern found")
                else:
                    raise HTTPException(status_code=404, detail="No detection files found")
            
        elif saved_hair_line_type == "FreeMark":
            if "mask_8months" not in mask_paths or not os.path.exists(mask_paths["mask_8months"]):
                raise HTTPException(status_code=404, detail="FreeMark 8-month mask not found")
            
            with open(mask_paths["mask_8months"], "rb") as f:
                mask_data = f.read()
            
            logger.info(f"GENERATE-8M-{request_id}: Loaded FreeMark 8-month mask ({len(mask_data)} bytes)")
            
            generated_image = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_data,
                "8months", hair_density_8m, hair_type, hair_color, 
                "FreeMark", request_id
            )
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
            else:
                # Fallback to coordinate generation
                coords = generate_hairline_coordinates_from_face_detection(
                    image_width, image_height, request_id, saved_pattern
                )
                
                # Note: This would need the generate_freemark_hair function if you want to support it
                # For now, assuming mask-based generation
                raise HTTPException(status_code=404, detail="Hairline mask not found")
        
        if not generated_image:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        return {
            "request_id": request_id,
            "image": base64.b64encode(generated_image).decode("utf-8"),
            "timeframe": "8months",
            "generation_mode": f"mask_based_{saved_hair_line_type.lower().replace(' ', '_')}",
            "pattern_type": saved_hair_line_type
        }
        
    except Exception as e:
        logger.error(f"GENERATE-8M-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"8-month generation failed: {str(e)}")
        
if __name__ == "__main__":
    logger.info("STARTUP: Hair Growth API with prompt.json Integration")
    uvicorn.run(app, host="0.0.0.0", port=8000)
