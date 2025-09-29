# Updated main.py with consolidated /individual endpoint
print("Starting main.py...")

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
    
    GEMINI_API_KEY = "AIzaSyARkGwZERa_Gzy_XwlyjcBw7-U02o4YKDg"  # Replace with your valid API key 
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY is required. Please set it in main.py.")
        raise ValueError("GEMINI_API_KEY is required. Please set it in main.py.")

    # Configure Gemini API before listing models
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        raise

    # Set a default model name (update after checking logs) 
    MODEL_NAME = "gemini-2.5-flash-image-preview"  # Use supported model for image generation

    logger.info("All imports successful")

    # Create FastAPI app
    app = FastAPI(title="Hair Growth Simulation API with Face Detection")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Configuration
    GENERATION_CONFIG = {
        "max_retries": 3,
        "quality_check_delay": 2,
        "generation_timeout": 120
    }
    
    # Global variable to store prompts config
    PROMPTS_CONFIG = None
    
    # Create necessary directories
    directories = ["image_logs", "uploads", "detections"]
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

def format_coordinates_for_prompt(coordinates_data: dict, request_id: str = "") -> str:
    """ENHANCED: Generate coordinate prompt with priority-based density multipliers for half-bald optimization"""
    if not coordinates_data or "regions" not in coordinates_data:
        return "[NO MARKINGS] Use general hair growth approach."
    
    regions = coordinates_data["regions"]
    if not regions:
        return "[NO REGIONS] Use general hair growth approach."
    
    logger.info(f"PROMPT-{request_id}: Formatting {len(regions)} enhanced regions with priority multipliers")
    
    # Calculate priority distribution for enhanced targeting
    priority_counts = coordinates_data.get("priority_distribution", {})
    total_area = coordinates_data.get("total_marked_area", 0)
    coverage_ratio = coordinates_data.get("coverage_ratio", 0)
    
    prompt_parts = [
        f"[ENHANCED COORDINATE-BASED HAIR GENERATION WITH PRIORITY MULTIPLIERS]",
        f"Generate DRAMATICALLY VISIBLE hair growth in EXACTLY {len(regions)} marked regions:",
        f"Total marked area: {total_area} pixels ({coverage_ratio:.1%} coverage)",
        f"Priority distribution: {priority_counts}",
        ""
    ]
    
    for i, region in enumerate(regions, 1):
        x, y, w, h = region["bbox"]
        center_x, center_y = region["center"]
        priority = region.get("density_priority", "medium")
        multiplier = region.get("density_multiplier", 1.0)
        area_ratio = region.get("area_ratio", 0)
        growth_dir = region.get("growth_direction", "natural")
        
        prompt_parts.extend([
            f"REGION {i} - {priority.upper()} PRIORITY (Area: {region['area']} pixels, {area_ratio:.2%} of image):",
            f"  • Boundaries: X={x} to {x+w}, Y={y} to {y+h}",
            f"  • Center point: ({center_x}, {center_y})",
            f"  • Density multiplier: {multiplier}x (enhanced visibility)",
            f"  • Growth direction: {growth_dir}",
            f"  • Visibility requirement: {'MAXIMUM DRAMATIC IMPACT' if priority == 'ultra_high' else 'HIGH DRAMATIC IMPACT' if priority == 'high' else 'CLEAR VISIBLE IMPROVEMENT' if priority == 'medium' else 'BASELINE IMPROVEMENT'}",
            ""
        ])
    
    prompt_parts.extend([
        "ENHANCED GENERATION REQUIREMENTS FOR HALF-BALD OPTIMIZATION:",
        "✓ ABSOLUTE BOUNDARY ENFORCEMENT: Generate hair ONLY within exact coordinate boundaries - ZERO EXCEPTIONS",
        "✓ PRIORITY-BASED ENHANCEMENT: Apply region-specific density multipliers for maximum visibility",
        "✓ AGGRESSIVE CONTRAST: Use optimal lighting and shadowing for maximum hair-scalp contrast",
        "✓ THICKNESS ENHANCEMENT: Generate visibly thicker strands for improved visibility in sparse areas",
        "✓ NATURAL INTEGRATION: Seamless blending with existing hair while maximizing visual impact",
        "✓ HALF-BALD OPTIMIZATION: Special attention to sparse hair regions with enhanced density",
        "✓ COORDINATE PRECISION: Every hair strand must originate within marked coordinate boundaries",
        ""
    ])
    
    return "\n".join(prompt_parts)

def build_freemark_generation_prompt(coordinates_data: dict, timeframe: str, 
                                   hair_density: float, hair_type: str, 
                                   hair_color: str, request_id: str = "") -> str:
    """FIXED: Enhanced prompt with stronger 3-month visibility requirements"""
    logger.info(f"PROMPT-{request_id}: Building ENHANCED prompt for {timeframe}")
    
    # Load prompts configuration
    global PROMPTS_CONFIG
    if not PROMPTS_CONFIG:
        PROMPTS_CONFIG = load_prompts_config()
    
    coord_prompt = format_coordinates_for_prompt(coordinates_data, request_id)
    
    # FIXED: More aggressive parameters for 3-month generation
    if timeframe in ["3months", "3 months", "3 Months"]:
        timeframe_key = "3months"
        base_multiplier = 1.25     # Realistic 3-month growth (25%)
        density_boost = 1.0        # No artificial boost for natural results
        timeframe_desc = f"SUBTLE EARLY REGROWTH with {hair_density * 100:.1f}% base density achieving 25% subtle growth"
        growth_type = "THICK, CLEARLY VISIBLE emerging hair strands with MAXIMUM early-stage impact"
        visibility_req = "IMMEDIATE_VISIBLE_TRANSFORMATION"
        maturity_level = "ENHANCED EARLY STAGE WITH DRAMATIC VISIBILITY"
    else:
        timeframe_key = "8months"
        base_multiplier = 1.5      # Realistic 8-month growth (50% improvement)
        density_boost = 1.0        # No artificial boost for natural results
        timeframe_desc = f"MAXIMUM MATURE GROWTH with {hair_density * 100:.1f}% base density + {base_multiplier}x COMPLETE transformation"
        growth_type = "EXTREMELY THICK, FULLY ESTABLISHED hair with TOTAL visual impact"
        visibility_req = "COMPLETE_TRANSFORMATION"
        maturity_level = "FULLY MATURE WITH MAXIMUM DENSITY"

    # Color and texture descriptions (unchanged)
    if hair_color == "#000000":
        color_desc = "DEEP JET BLACK hair with natural highlights and dramatic scalp contrast"
    elif hair_color == "#8B4513":
        color_desc = "RICH DARK BROWN hair with natural depth and warm undertones"
    elif hair_color == "#654321":
        color_desc = "MEDIUM BROWN hair with natural variation and realistic coloring"
    elif hair_color == "#D2B48C":
        color_desc = "WARM LIGHT BROWN/TAN hair with natural golden highlights"
    elif hair_color == "#FFD700":
        color_desc = "GOLDEN BLONDE hair with natural shine and realistic tones"
    elif hair_color == "#C0C0C0":
        color_desc = "DISTINGUISHED SILVER-GREY hair with natural variation"
    else:
        color_desc = f"NATURAL hair color matching {hair_color} with realistic depth and shine"
    
    if hair_type.lower() == "straight hair":
        texture_desc = "STRAIGHT, SLEEK hair texture flowing naturally with smooth alignment"
    elif hair_type.lower() == "wavy hair":
        texture_desc = "NATURAL WAVY hair texture with gentle curves and organic movement"
    elif hair_type.lower() == "curly hair":
        texture_desc = "CURLY hair texture with defined spirals and natural volume"
    elif hair_type.lower() == "coily hair":
        texture_desc = "COILY hair texture with tight natural curls and authentic volume"
    else:
        texture_desc = f"NATURAL {hair_type.upper()} texture with authentic characteristics"
    
    effective_density = hair_density * base_multiplier * density_boost
    effective_percentage = min(100, effective_density * 100)
    
    # FIXED: Enhanced prompt with stronger 3-month requirements
    prompt = f"""PRECISE HAIR RESTORATION - {maturity_level}

MISSION: Create spiky, short-length hair growth matching existing style

{coord_prompt}

🌱 {timeframe_key.upper()} STYLE SPECIFICATIONS:
✅ STYLE TYPE: Short, spiky hair with natural lift
✅ DENSITY: {hair_density * 100:.1f}% coverage distributed like reference
✅ LENGTH: Short to medium-short length (1-2 inches)
✅ DIRECTION: Forward and slightly downward flow
✅ TEXTURE: Spiky, textured appearance with natural lift

🎯 PRECISE STYLE MATCH:
• STYLING: Short, spiky texture with natural volume
• COLOR: {color_desc} matching existing hair exactly
• TEXTURE: Individual spiky strands with natural lift
• MOVEMENT: Forward-flowing with slight downward angle
• DISTRIBUTION: Natural coverage focusing on hairline and temples

⚡ STYLE REQUIREMENTS:
1. SPIKY TEXTURE: Create distinct, spiky strands like reference
2. NATURAL LIFT: Add slight volume and lift at hairline
3. FORWARD FLOW: Maintain forward-angled hair direction
4. PRECISE LENGTH: Keep hair short (1-2 inches) throughout
5. EXACT COLOR: Match existing {color_desc} perfectly
6. STYLE MATCH: Replicate spiky, textured look of reference
7. HAIRLINE FOCUS: Emphasize natural M-shaped pattern
8. {"EARLY TEXTURE" if timeframe == "3months" else "FULL STYLE"}: Show {timeframe} progression

STYLE EXECUTION:
- Generate short, spiky hair texture
- Maintain {effective_percentage:.0f}% density with spiky distribution
- Create forward-angled hair direction
- Match existing hair color exactly
- Focus on natural M-shaped hairline pattern
- Ensure consistent short length (1-2 inches)
- Add natural lift and volume at hairline
- Keep individual spiky strands visible"""

    logger.info(f"PROMPT-{request_id}: Generated ENHANCED {timeframe} prompt")
    logger.info(f"  - Effective Density: {effective_percentage:.0f}% (Enhanced multipliers)")
    logger.info(f"  - Visibility: {visibility_req}")
    
    return prompt

def extract_image_from_response_simplified(response, request_id: str = ""):
    """SIMPLIFIED: Extract image data from Gemini response"""
    logger.info(f"EXTRACT-{request_id}: Extracting image from response")
    
    try:
        # Method 1: Direct parts access
        if hasattr(response, 'parts') and response.parts:
            for i, part in enumerate(response.parts):
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'data'):
                        return decode_image_data(part.inline_data.data, f"{request_id}-parts-{i}")
        
        # Method 2: Candidates access
        if hasattr(response, 'candidates') and response.candidates:
            for j, candidate in enumerate(response.candidates):
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        for k, part in enumerate(candidate.content.parts):
                            if hasattr(part, 'inline_data') and part.inline_data:
                                if hasattr(part.inline_data, 'data'):
                                    return decode_image_data(part.inline_data.data, f"{request_id}-candidate-{j}-{k}")
        
        # Method 3: Direct response data
        if hasattr(response, 'data'):
            return decode_image_data(response.data, f"{request_id}-direct")
        
        logger.warning(f"EXTRACT-{request_id}: No image data found in response structure")
        
        # Debug: Log response structure
        logger.debug(f"EXTRACT-{request_id}: Response type: {type(response)}")
        logger.debug(f"EXTRACT-{request_id}: Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
        
        return None
        
    except Exception as e:
        logger.error(f"EXTRACT-{request_id}: Error extracting image: {str(e)}")
        return None

def decode_image_data(data, context: str = ""):
    """FIXED: Decode image data handling both string and bytes"""
    try:
        if isinstance(data, str):
            decoded = base64.b64decode(data)
            logger.info(f"DECODE-{context}: Decoded base64 string to {len(decoded)} bytes")
            return decoded
        elif isinstance(data, bytes):
            # Already bytes
            logger.info(f"DECODE-{context}: Using direct bytes data ({len(data)} bytes)")
            return data
        else:
            logger.warning(f"DECODE-{context}: Unknown data type: {type(data)}")
            return None
    except Exception as e:
        logger.error(f"DECODE-{context}: Decode error: {str(e)}")
        return None

async def generate_freemark_hair(input_image_data: bytes, input_mime_type: str,
                                coordinates_data: dict, timeframe: str, 
                                hair_density: float, hair_type: str, 
                                hair_color: str, request_id: str = ""):
    """IMPROVED: More reliable FreeMark generation with better error handling"""
    logger.info(f"GEN-{request_id}: Starting IMPROVED FreeMark generation")
    
    try:
        # Build prompt with parameter validation
        if hair_density <= 0 or hair_density > 1:
            logger.warning(f"GEN-{request_id}: Invalid density {hair_density}, clamping to 0.1-1.0")
            hair_density = max(0.1, min(1.0, hair_density))
        
        prompt = build_freemark_generation_prompt(coordinates_data, timeframe, hair_density, hair_type, hair_color, request_id)
        
        # Enhanced logging for debugging
        logger.info(f"GEN-{request_id}: Generation parameters:")
        logger.info(f"  - Hair Color: {hair_color}")
        logger.info(f"  - Hair Type: {hair_type}")  
        logger.info(f"  - Hair Density: {hair_density * 100:.1f}%")
        logger.info(f"  - Timeframe: {timeframe}")
        logger.info(f"  - Prompt length: {len(prompt)} characters")
        image_b64 = base64.b64encode(input_image_data).decode('utf-8')
        
        # Single, more reliable content format
        content = [
            {
                "inline_data": {
                    "mime_type": input_mime_type,
                    "data": image_b64
                }
            },
            prompt
        ]
        
        # IMPROVED: Try models in order of reliability
        models_to_try = ["gemini-2.5-flash-image-preview"
        ]
        
        for model_name in models_to_try:
            logger.info(f"GEN-{request_id}: Attempting generation with {model_name}")
            
            try:
                model = genai.GenerativeModel(model_name)
                
                # IMPROVED: Add generation config for consistency
                generation_config = {
                    "temperature": 0.4,  # Lower temperature for more consistent results
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                }
                
                # Generate with timeout and config
                response = await asyncio.wait_for(
                    asyncio.to_thread(model.generate_content, content, generation_config=generation_config),
                    timeout=60  # 60 second timeout
                )
                
                # IMPROVED: Better response validation
                if not response:
                    logger.warning(f"GEN-{request_id}: Empty response from {model_name}")
                    continue
                    
                # Extract image data with improved error handling
                image_data = extract_image_from_response_simplified(response, request_id)
                
                if image_data:
                    # IMPROVED: Validate generated image
                    if len(image_data) < 5000:  # Too small to be a valid image
                        logger.warning(f"GEN-{request_id}: Generated image too small ({len(image_data)} bytes)")
                        continue
                    
                    logger.info(f"GEN-{request_id}: ✓ SUCCESS with {model_name}")
                    logger.info(f"  - Generated image size: {len(image_data)/1024:.1f}KB")
                    return image_data
                else:
                    logger.warning(f"GEN-{request_id}: No image data extracted from {model_name}")
                    continue
                        
            except asyncio.TimeoutError:
                logger.error(f"GEN-{request_id}: Timeout with {model_name}")
                continue
            except Exception as model_error:
                logger.warning(f"GEN-{request_id}: Model {model_name} failed: {str(model_error)}")
                continue
        
        logger.error(f"GEN-{request_id}: All models failed to generate")
        return None
        
    except Exception as e:
        logger.error(f"GEN-{request_id}: Critical error in generation: {str(e)}")
        return None
    
def build_mask_based_hairline_prompt(timeframe: str, hair_density: float, hair_type: str, 
                                   hair_color: str, pattern_type: str, request_id: str = "",
                                   input_image_size: int = 0, mask_image_size: int = 0, 
                                   white_pixel_count: int = 0, total_pixels: int = 0) -> str:
    """FIXED: Use exact density values for Crown/Full Scalp/Mid Crown without modifications"""
    logger.info(f"MASK-PROMPT-{request_id}: Building prompt for {pattern_type} - {timeframe}")
    
    try:
        hair_density = max(0.1, min(1.0, hair_density))
        
        # FIXED: Use exact input density with correct timeframe progression
        if timeframe in ["3months", "3 months"]:
            # For 3 months: Increase input density by 25% (realistic early growth)
            enhancement_multiplier = 1.25    # Realistic 25% increase from input
            density_multiplier = 1.0        # No additional boost
            maturity_desc = "EARLY GROWTH STAGE"
            result_desc = "emerging hair with natural early growth characteristics"
            visibility_requirement = "NATURAL_EARLY_GROWTH"
            strand_requirement = "Natural early-stage hair development"
            color_variation = "slightly lighter new growth color intensity"
        else:
            # For 8 months: More realistic growth progression
            
            enhancement_multiplier = 1.38   # More realistic 8-month growth
            density_multiplier = 1.0        # No additional boost
            maturity_desc = "MATURE GROWTH STAGE"
            result_desc = "fully established hair with complete maturity"
            visibility_requirement = "MATURE_COMPLETE_GROWTH" 
            strand_requirement = "Full mature hair development"
            color_variation = "full mature hair color intensity"

        # Enhanced color mapping with timeframe variations
        if pattern_type in ["Crown", "Mid Crown", "Full Scalp"]:
            color_mapping = {
                "#000000": f"natural black hair with {color_variation}",
                "#8B4513": f"natural dark brown hair with {color_variation}",
                "#654321": f"natural medium brown hair with {color_variation}",
                "#D2B48C": f"natural light brown hair with {color_variation}",
                "#FFD700": f"natural blonde hair with {color_variation}",
                "#808080": f"natural grey hair with {color_variation}"
            }
        else:
            color_mapping = {
                "#000000": "ULTRA-DEEP JET BLACK with MAXIMUM contrast and dramatic definition",
                "#8B4513": "ULTRA-RICH DARK BROWN with intense depth and maximum visibility",
                "#654321": "ENHANCED MEDIUM BROWN with strong contrast and clear definition",
                "#D2B48C": "BRIGHT GOLDEN BROWN with maximum shine and visibility",
                "#FFD700": "ULTRA-BRIGHT GOLDEN BLONDE with intense shine and contrast",
                "#808080": "ULTRA-ENHANCED SILVER-GREY with MAXIMUM contrast and dramatic depth"
            }
        
        color_desc = color_mapping.get(hair_color, f"natural {hair_color} hair with {color_variation}")

        # Enhanced texture mapping
        if pattern_type in ["Crown", "Mid Crown", "Full Scalp"]:
            texture_mapping = {
                "straight hair": f"natural straight hair texture with {maturity_desc.lower()} characteristics",
                "wavy hair": f"natural wavy hair texture with {maturity_desc.lower()} wave definition",
                "curly hair": f"natural curly hair texture with {maturity_desc.lower()} curl formation",
                "coily hair": f"natural coily hair texture with {maturity_desc.lower()} coil structure"
            }
        else:
            texture_mapping = {
                "straight hair": "ULTRA-STRAIGHT with maximum smoothness and dramatic scalp contrast",
                "wavy hair": "ULTRA-WAVY with pronounced curves and maximum volume visibility", 
                "curly hair": "ULTRA-CURLY with dramatic spirals and maximum dimensional presence",
                "coily hair": "ULTRA-COILY with intense curl patterns and maximum textural visibility"
            }
        
        hair_texture_desc = texture_mapping.get(hair_type.lower(), f"natural {hair_type.upper()}")

        # Calculate effective density - EXACT for Crown patterns, enhanced for others
        effective_density = hair_density * enhancement_multiplier * density_multiplier
        effective_percentage = min(100, effective_density * 100)

        # FIXED: Different prompt styles based on pattern type
        if pattern_type in ["Crown", "Mid Crown", "Full Scalp"]:
            complete_prompt = f"""PROFESSIONAL HAIR RESTORATION - {maturity_desc} FOR {pattern_type.upper()}

🎯 MISSION: Generate natural {timeframe} {pattern_type.lower()} hair restoration

MASK-BASED GENERATION:
- Image 1: Patient photo requiring {pattern_type.lower()} hair restoration
- Image 2: Mask with BLACK background and WHITE areas = exact hair generation zones
- Generate hair ONLY in WHITE mask areas with natural appearance

EXACT PARAMETERS:
• HAIR COLOR: {color_desc}
  → Apply natural {hair_color} coloring throughout
  
• HAIR TEXTURE: {hair_texture_desc}
  → Maintain consistent texture pattern
  
• HAIR DENSITY: {hair_density * 100:.1f}% coverage (exact frontend value)
  → Apply uniform density across white areas
  
• TIMEFRAME: {timeframe} {maturity_desc.lower()}
  → Show appropriate {timeframe} development stage

GENERATION REQUIREMENTS:
1. MASK COMPLIANCE: Generate hair ONLY in white areas - respect boundaries
2. DENSITY ACCURACY: Apply exactly {hair_density * 100:.1f}% hair coverage
3. COLOR CONSISTENCY: Use {color_desc} throughout
4. TEXTURE UNIFORMITY: Apply {hair_texture_desc} pattern
5. NATURAL APPEARANCE: Show realistic {timeframe} hair characteristics

EXECUTE: Generate natural {timeframe} {pattern_type} restoration."""

        else:
            # Keep existing ultra-enhanced prompt for other patterns
            complete_prompt = f"""ULTRA-PROFESSIONAL HAIR RESTORATION - EXTREME VISIBILITY ENFORCEMENT

🎯 ULTRA-EXTREME MISSION: Generate {visibility_requirement} with IMPOSSIBLE-TO-MISS results

CRITICAL ANALYSIS:
- Image 1: Patient requiring DRAMATIC hair restoration
- Image 2: WHITE mask = EXTREME HAIR GENERATION ZONES (200% effort required)
- Pattern: {pattern_type} with ULTRA-ENHANCED characteristics
- Target: {maturity_desc} with ZERO subtlety allowed

🚨 EXTREME PARAMETER REQUIREMENTS:
• HAIR COLOR: {color_desc}
  → EVERY HAIR STRAND must be DRAMATICALLY VISIBLE
  → MAXIMUM color saturation and contrast required
  
• HAIR TEXTURE: {hair_texture_desc}
  → ALL hair must be EXTREMELY PRONOUNCED
  → MAXIMUM texture definition required
  
• HAIR DENSITY: {effective_percentage:.0f}% EXTREME coverage
  → Base: {hair_density * 100:.1f}% × {enhancement_multiplier}x × {density_multiplier}x = {effective_percentage:.0f}%
  → MINIMUM {effective_percentage:.0f}% coverage - NO COMPROMISES
  
• TIMEFRAME: {timeframe} with EXTREME early visibility
  → Show IMPOSSIBLE-TO-MISS development stage

🔥 ULTRA-EXTREME GENERATION RULES:
1. VISIBILITY ENFORCEMENT: {strand_requirement}
2. CONTRAST ENFORCEMENT: Use MAXIMUM lighting contrast for visibility
3. THICKNESS ENFORCEMENT: Generate ULTRA-THICK strands (3x normal thickness)
4. COLOR ENFORCEMENT: Apply {color_desc} with MAXIMUM saturation
5. BOUNDARY ENFORCEMENT: Generate ONLY in WHITE areas with EXTREME density
6. QUALITY ENFORCEMENT: Result must be DRAMATICALLY OBVIOUS

EXECUTE: Generate {timeframe} {pattern_type} with EXTREME VISIBILITY."""

        logger.info(f"MASK-PROMPT-{request_id}: Generated {pattern_type} prompt for {timeframe}")
        logger.info(f"  - Density: {hair_density * 100:.1f}% (frontend) → {effective_percentage:.0f}% (effective)")
        logger.info(f"  - Enhancement: {enhancement_multiplier}x × {density_multiplier}x")
        
        return complete_prompt
        
    except Exception as e:
        logger.error(f"MASK-PROMPT-{request_id}: Error: {str(e)}")
        return f"Generate natural hair restoration for {pattern_type} at {timeframe}"
    
@app.post("/generate-with-saved-pattern")
async def generate_with_saved_pattern(
    image: UploadFile = File(...),
    hair_color: str = Form("#000000"),
    hair_type: str = Form("Straight Hair"),
    hair_line_type: str = Form("Hairline"),
    hair_density_3m: float = Form(0.7),
    hair_density_8m: float = Form(0.9),
    timeframe: str = Form("3months"),
    face_detected: str = Form("false"),
    use_saved_pattern: str = Form("true")
):
    """FIXED: Generate hair using previously saved pattern/mask data with correct pattern handling"""
    request_id = generate_request_id()
    logger.info(f"GENERATE-SAVED-{request_id}: Generating hair using saved pattern data")
    
    try:
        # Read input image
        input_data = await image.read()
        input_image = Image.open(io.BytesIO(input_data))
        image_width, image_height = input_image.size
        
        # Find and load saved settings
        settings_dir = os.path.join("logs", "saved_settings")
        if not os.path.exists(settings_dir):
            raise HTTPException(status_code=404, detail="No saved settings found. Please save settings first.")
        
        # Get the most recent settings file
        settings_files = [f for f in os.listdir(settings_dir) if f.endswith("_settings.json")]
        if not settings_files:
            raise HTTPException(status_code=404, detail="No saved settings found. Please save settings first.")
        
        settings_files_with_time = []
        for f in settings_files:
            file_path = os.path.join(settings_dir, f)
            mod_time = os.path.getmtime(file_path)
            settings_files_with_time.append((mod_time, f))
        
        latest_settings_file = sorted(settings_files_with_time)[-1][1]
        settings_path = os.path.join(settings_dir, latest_settings_file)
        
        with open(settings_path, "r") as f:
            saved_settings = json.load(f)
        
        saved_hair_line_type = saved_settings.get("hair_line_type", "Hairline")
        logger.info(f"  - Saved hair line type: {saved_hair_line_type}")
        
        if saved_hair_line_type == "Hairline":
            # FIXED: Proper hairline pattern handling with actual saved pattern
            saved_hairline_pattern = saved_settings.get("hairline_pattern", "M_pattern")
            logger.info(f"  - Using HAIRLINE generation with pattern: {saved_hairline_pattern}")
            
            mask_paths = saved_settings.get("mask_paths", {})
            
            if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
                # Use saved hairline mask
                with open(mask_paths["hairline_mask"], "rb") as f:
                    hairline_mask_data = f.read()
                
                generated_image_3m = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, hairline_mask_data,
                    "3months", hair_density_3m, hair_type, hair_color, saved_hairline_pattern, f"{request_id}-3m"
                )
                
                generated_image_8m = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, hairline_mask_data,
                    "8months", hair_density_8m, hair_type, hair_color, saved_hairline_pattern, f"{request_id}-8m"
                )
            else:
                # Fallback to coordinate-based generation with correct pattern
                coordinates_data = generate_hairline_coordinates_from_face_detection(
                    image_width=image_width,
                    image_height=image_height,
                    request_id=request_id,
                    pattern_type=saved_hairline_pattern  # FIXED: Use actual saved pattern
                )
                
                generated_image_3m = await generate_freemark_hair(
                    input_data, image.content_type, coordinates_data,
                    "3months", hair_density_3m, hair_type, hair_color, f"{request_id}-3m"
                )
                
                generated_image_8m = await generate_freemark_hair(
                    input_data, image.content_type, coordinates_data,
                    "8months", hair_density_8m, hair_type, hair_color, f"{request_id}-8m"
                )
            
            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_image_3m if timeframe == "3months" else generated_image_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_image_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_image_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": f"hairline_{saved_hairline_pattern}",
                "used_saved_pattern": True,
                "pattern_type": saved_hairline_pattern
            }
        elif saved_hair_line_type in ["Crown", "Mid Crown", "Full Scalp"]:
            logger.info(f"  - Using MASK-BASED generation for {saved_hair_line_type}")
            
            # Find the corresponding pattern mask from face detection
            mask_paths = saved_settings.get("mask_paths", {})
            
            # Look for the saved pattern mask
            pattern_mask_path = None
            if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
                pattern_mask_path = mask_paths["hairline_mask"]
                logger.info(f"  - Found saved {saved_hair_line_type} mask: {pattern_mask_path}")
            else:
                # Map hair line types to pattern keys for file search
                pattern_key_mapping = {
                    "Crown": "crown",
                    "Mid Crown": "mid_crown", 
                    "Full Scalp": "full_scalp"
                }
                
                pattern_key = pattern_key_mapping.get(saved_hair_line_type)
                
                # Try to find pattern from face detection results
                if os.path.exists("detections"):
                    pattern_files = [f for f in os.listdir("detections") if pattern_key in f.lower()]
                    if pattern_files:
                        latest_pattern = max(pattern_files, key=lambda x: os.path.getmtime(os.path.join("detections", x)))
                        pattern_file_path = os.path.join("detections", latest_pattern)
                        
                        # Convert face detection pattern to FreeMark style mask
                        with open(pattern_file_path, "rb") as f:
                            pattern_data = f.read()
                        
                        pattern_mask_data = convert_to_freemark_style_mask(pattern_data, saved_hair_line_type, request_id)
                        logger.info(f"  - Converted {saved_hair_line_type} pattern to mask")
                    else:
                        raise HTTPException(status_code=404, detail=f"No saved mask found for {saved_hair_line_type} pattern")
                else:
                    raise HTTPException(status_code=404, detail=f"No detections directory found for {saved_hair_line_type} pattern")
            
            if pattern_mask_path:
                with open(pattern_mask_path, "rb") as f:
                    pattern_mask_data = f.read()
            
            # Generate both timeframes using mask-based approach with EXACT density
            logger.info(f"  - Generating {saved_hair_line_type} for BOTH timeframes with exact density...")
            
            generated_image_3m = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, pattern_mask_data,
                "3months", hair_density_3m, hair_type, hair_color, saved_hair_line_type, f"{request_id}-3m"
            )
            
            generated_image_8m = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, pattern_mask_data,
                "8months", hair_density_8m, hair_type, hair_color, saved_hair_line_type, f"{request_id}-8m"
            )
            
            if not generated_image_3m or not generated_image_8m:
                raise HTTPException(status_code=500, detail=f"Failed to generate {saved_hair_line_type} images")
            
            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_image_3m if timeframe == "3months" else generated_image_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_image_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_image_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": f"mask_based_{saved_hair_line_type.lower().replace(' ', '_')}",
                "used_saved_pattern": True,
                "pattern_type": saved_hair_line_type
            }
        else:
            # FreeMark mode
            logger.info(f"  - Using FREEMARK generation")
            mask_paths = saved_settings.get("mask_paths", {})
            
            if "mask_3months" not in mask_paths or "mask_8months" not in mask_paths:
                missing = []
                if "mask_3months" not in mask_paths: missing.append("3months")
                if "mask_8months" not in mask_paths: missing.append("8months")
                raise HTTPException(status_code=404, detail=f"Missing saved masks for: {', '.join(missing)}")
            
            # Generate both timeframes
            with open(mask_paths["mask_3months"], "rb") as f:
                mask_3m_data = f.read()
            with open(mask_paths["mask_8months"], "rb") as f:
                mask_8m_data = f.read()
            
            generated_image_3m = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_3m_data,
                "3months", hair_density_3m, hair_type, hair_color, "FreeMark", f"{request_id}-3m"
            )
            
            generated_image_8m = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_8m_data,
                "8months", hair_density_8m, hair_type, hair_color, "FreeMark", f"{request_id}-8m"
            )
            
            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_image_3m if timeframe == "3months" else generated_image_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_image_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_image_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": "freemark_dual_masks",
                "used_saved_pattern": True,
                "pattern_type": "FreeMark"
            }
        
    except Exception as e:
        logger.error(f"GENERATE-SAVED-{request_id}: Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def build_freemark_generation_prompt(coordinates_data: dict, timeframe: str, 
                                   hair_density: float, hair_type: str, 
                                   hair_color: str, request_id: str = "") -> str:
    """ULTRA-ENHANCED FreeMark prompt with extreme 3-month visibility"""
    logger.info(f"FREEMARK-ULTRA-{request_id}: Building EXTREME FreeMark prompt for {timeframe}")
    
    global PROMPTS_CONFIG
    if not PROMPTS_CONFIG:
        PROMPTS_CONFIG = load_prompts_config()
    
    coord_prompt = format_coordinates_for_prompt(coordinates_data, request_id)
    
    # Hair density progression with realistic early growth
    if timeframe in ["3months", "3 months"]:
        # For 3 months: Show subtle initial growth
        base_multiplier = 1.25      # Realistic 3-month growth
        density_boost = 1.0         # No additional boost
        visibility_req = "SUBTLE_INITIAL_GROWTH"
        maturity_level = "EARLY REGROWTH PHASE"
        growth_type = "Fine, short initial hair growth with subtle appearance"
    else:
        # For 8 months: Show fuller established growth
        base_multiplier = 1.25       # Realistic 8-month growth
        density_boost = 1.0         # No additional boost
        visibility_req = "COMPLETE_COVERAGE"
        maturity_level = "FULL MATURE GROWTH"
        growth_type = "Established, fuller hair coverage with natural thickness"

    # Enhanced color descriptions
    color_mapping = {
        "#000000": "ULTRA-DRAMATIC JET BLACK with MAXIMUM scalp contrast",
        "#808080": "ULTRA-ENHANCED SILVER-GREY with EXTREME visibility and depth"
    }
    color_desc = color_mapping.get(hair_color, f"ULTRA-ENHANCED {hair_color}")
    
    texture_mapping = {
        "wavy hair": "ULTRA-PRONOUNCED WAVY texture with MAXIMUM dimensional visibility"
    }
    texture_desc = texture_mapping.get(hair_type.lower(), f"ULTRA-ENHANCED {hair_type}")
    
    effective_density = hair_density * base_multiplier * density_boost
    effective_percentage = min(100, effective_density * 100)
    
    # ULTRA-ENHANCED FreeMark prompt
    prompt = f"""ULTRA-EXPERT HAIR RESTORATION - {maturity_level}

ULTRA-EXTREME MISSION: Generate {visibility_req} with COORDINATE PRECISION

{coord_prompt}

🚀 ULTRA-ENHANCED {timeframe.upper()} SPECIFICATIONS:
✅ MATURITY: {maturity_level}
✅ EXTREME DENSITY: {hair_density * 100:.1f}% × {base_multiplier}x × {density_boost}x = {effective_percentage:.0f}%
✅ VISIBILITY: {visibility_req} - MUST BE DRAMATICALLY OBVIOUS
✅ STRAND TYPE: {growth_type}

🎯 ULTRA-EXTREME CHARACTERISTICS:
• COLOR: {color_desc} (MAXIMUM saturation required)
• TEXTURE: {texture_desc} (EXTREME definition required)
• DENSITY: {effective_percentage:.0f}% (MINIMUM acceptable coverage)
• VISIBILITY: IMPOSSIBLE TO MISS from any angle

⚡ ULTRA-CRITICAL REQUIREMENTS:
1. EXTREME VISIBILITY: Every hair must be CLEARLY VISIBLE
2. MAXIMUM CONTRAST: Use optimal lighting for DRAMATIC distinction
3. ULTRA-THICKNESS: Apply {density_boost}x thickness multiplication
4. PERFECT BOUNDARIES: Generate ONLY within coordinate areas
5. DRAMATIC IMPACT: Result must be IMMEDIATELY NOTICEABLE

EXECUTE: Generate {timeframe} restoration with EXTREME VISIBILITY."""

    logger.info(f"FREEMARK-ULTRA-{request_id}: EXTREME prompt ready")
    logger.info(f"  - Ultra Density: {effective_percentage:.0f}%")
    
    return prompt

def convert_to_freemark_style_mask(pattern_image_data: bytes, request_id: str = "", pattern_type: str = "M Pattern") -> bytes:
    """Convert pattern image to FreeMark style mask based on pattern type"""
    logger.info(f"FREEMARK-MASK-{request_id}: Converting {pattern_type} to FreeMark style mask")
    
    try:
        # Load pattern image
        pattern_image = Image.open(io.BytesIO(pattern_image_data))
        width, height = pattern_image.size
        logger.info(f"FREEMARK-MASK-{request_id}: Original pattern size: {width}x{height}")
        
        # Convert to grayscale for processing
        gray_pattern = pattern_image.convert('L')
        pattern_array = np.array(gray_pattern)
        
        # Create FreeMark style mask: black background, white areas
        freemark_mask = np.zeros((height, width, 3), dtype=np.uint8)  # Start with black
        
        # Pattern-specific processing
        if pattern_type.lower() in ['crown', 'full scalp', 'mid crown', 'full_scalp', 'mid_crown']:
            logger.info(f"FREEMARK-MASK-{request_id}: Processing {pattern_type} as RADIUS-based pattern (INNER WHITE)")
            
            # For crown/scalp patterns: create clean white INNER radius areas
            # Find the center of the pattern
            pattern_pixels = pattern_array > 30
            if np.any(pattern_pixels):
                # Find bounding box of pattern
                coords = np.where(pattern_pixels)
                min_y, max_y = np.min(coords[0]), np.max(coords[0])
                min_x, max_x = np.min(coords[1]), np.max(coords[1])
                
                center_y = (min_y + max_y) // 2
                center_x = (min_x + max_x) // 2
                
                # Calculate radius based on pattern size - make it generous for good coverage
                radius_y = (max_y - min_y) // 2 + 30  # More padding for better coverage
                radius_x = (max_x - min_x) // 2 + 30  # More padding for better coverage
                
                # Ensure minimum radius for visibility
                min_radius = min(width, height) // 8
                radius_y = max(radius_y, min_radius)
                radius_x = max(radius_x, min_radius)
                
                # Create elliptical/circular WHITE INNER area (hair growth region)
                y_coords, x_coords = np.ogrid[:height, :width]
                ellipse_mask = ((x_coords - center_x) / radius_x)**2 + ((y_coords - center_y) / radius_y)**2 <= 1
                
                freemark_mask[ellipse_mask] = [255, 255, 255]  # WHITE INNER area for hair growth
                
                logger.info(f"FREEMARK-MASK-{request_id}: Created WHITE INNER circle at ({center_x}, {center_y}) with radius ({radius_x}, {radius_y})")
            else:
                # Fallback: center circular WHITE INNER area
                center_x, center_y = width // 2, height // 2
                radius = min(width, height) // 3  # Larger radius for better coverage
                y_coords, x_coords = np.ogrid[:height, :width]
                circle_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                freemark_mask[circle_mask] = [255, 255, 255]  # WHITE INNER circle
                logger.info(f"FREEMARK-MASK-{request_id}: Created fallback WHITE INNER circle with radius {radius}")
                
        else:
            logger.info(f"FREEMARK-MASK-{request_id}: Processing {pattern_type} as SHAPE-based pattern (M/Z/Curve)")
            
            # For M/Z/Curve patterns: convert shape to solid white regions
            # Find pattern areas (non-black pixels)
            threshold = 30
            pattern_pixels = pattern_array > threshold
                        
            # Clean up small points and noise
            cleaned_pattern = ndimage.binary_opening(pattern_pixels, iterations=2)
            
            # Dilate to create solid regions (remove gaps between points)
            solid_pattern = ndimage.binary_dilation(cleaned_pattern, iterations=8)
            
            # Fill holes to create solid white regions
            filled_pattern = ndimage.binary_fill_holes(solid_pattern)
            
            # Apply final smoothing
            final_pattern = ndimage.binary_closing(filled_pattern, iterations=3)
            
            # Make pattern areas white
            freemark_mask[final_pattern] = [255, 255, 255]  # White pattern areas
            
            logger.info(f"FREEMARK-MASK-{request_id}: Created solid shape regions (removed points)")
        
        # Convert to PIL Image
        final_mask = Image.fromarray(freemark_mask)
        
        # Save to bytes
        mask_bytes = io.BytesIO()
        final_mask.save(mask_bytes, format='PNG')
        result_data = mask_bytes.getvalue()
        
        # Calculate statistics
        white_pixels = np.sum(freemark_mask[:,:,0] == 255)
        total_pixels = width * height
        white_percentage = (white_pixels / total_pixels) * 100
        
        logger.info(f"FREEMARK-MASK-{request_id}: Created {pattern_type} FreeMark mask ({len(result_data)} bytes)")
        logger.info(f"FREEMARK-MASK-{request_id}: White areas: {white_pixels} / {total_pixels} pixels ({white_percentage:.1f}%)")
        logger.info(f"FREEMARK-MASK-{request_id}: Black areas: {total_pixels - white_pixels} pixels ({100-white_percentage:.1f}%)")
        
        return result_data
        
    except Exception as e:
        logger.error(f"FREEMARK-MASK-{request_id}: Error converting {pattern_type}: {str(e)}")
        # Fallback based on pattern type
        try:
            pattern_image = Image.open(io.BytesIO(pattern_image_data))
            width, height = pattern_image.size
            
            fallback_mask = np.zeros((height, width, 3), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            
            if pattern_type.lower() in ['crown', 'full scalp', 'mid crown']:
                # Circular fallback for crown patterns
                radius = min(width, height) // 3
                y, x = np.ogrid[:height, :width]
                mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                fallback_mask[mask_circle] = [255, 255, 255]
                logger.info(f"FREEMARK-MASK-{request_id}: Used circular fallback for {pattern_type}")
            else:
                # Rectangular fallback for hairline patterns
                rect_width = width // 3
                rect_height = height // 6
                y1, y2 = center_y - rect_height, center_y + rect_height
                x1, x2 = center_x - rect_width, center_x + rect_width
                fallback_mask[y1:y2, x1:x2] = [255, 255, 255]
                logger.info(f"FREEMARK-MASK-{request_id}: Used rectangular fallback for {pattern_type}")
            
            final_mask = Image.fromarray(fallback_mask)
            mask_bytes = io.BytesIO()
            final_mask.save(mask_bytes, format='PNG')
            
            return mask_bytes.getvalue()
            
        except Exception as fallback_error:
            logger.error(f"FREEMARK-MASK-{request_id}: Fallback also failed: {str(fallback_error)}")
            return pattern_image_data
        
def convert_pattern_to_white_mask(pattern_image_data: bytes, request_id: str = "") -> bytes:
    """Convert pattern image to white mask for better generation (used during generation)"""
    logger.info(f"MASK-CONVERT-{request_id}: Converting pattern image to white mask")
    logger.info(f"  - Input mask size: {len(pattern_image_data)} bytes")
    
    try:
        # Load pattern image
        pattern_image = Image.open(io.BytesIO(pattern_image_data))
        logger.info(f"MASK-CONVERT-{request_id}: Original pattern size: {pattern_image.size}")
        
        # Convert to grayscale
        gray_pattern = pattern_image.convert('L')
        
        # Convert to numpy array for processing
        pattern_array = np.array(gray_pattern)
        
        # Create white mask where pattern exists (non-black areas)
        # ENHANCED: More aggressive thresholding for clearer mask
        threshold = 50  # Increased threshold for better detection
        white_mask = np.where(pattern_array > threshold, 255, 0).astype(np.uint8)
    
        # Dilate to fill small gaps
        kernel = np.ones((3,3), np.uint8)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        # Erode to restore original size
        white_mask = cv2.erode(white_mask, kernel, iterations=1)
        
        # Convert back to PIL Image
        mask_image = Image.fromarray(white_mask, mode='L')
        
        # Convert to RGB for consistency
        mask_rgb = Image.new('RGB', mask_image.size, (0, 0, 0))  # Black background
        mask_rgb.paste(mask_image, mask=mask_image)
        
        # Make pattern areas white
        mask_array = np.array(mask_rgb)
        mask_array[white_mask == 255] = [255, 255, 255]  # White where pattern exists
        
        final_mask = Image.fromarray(mask_array)
        
        # Save to bytes
        mask_bytes = io.BytesIO()
        final_mask.save(mask_bytes, format='PNG')
        result_data = mask_bytes.getvalue()
        
        logger.info(f"MASK-CONVERT-{request_id}: Converted to white mask ({len(result_data)} bytes)")
        logger.info(f"MASK-CONVERT-{request_id}: White pixels: {np.sum(white_mask == 255)} / {white_mask.size} total")
        
        return result_data
        
    except Exception as e:
        logger.error(f"MASK-CONVERT-{request_id}: Error converting pattern: {str(e)}")
        # Return original pattern if conversion fails
        return pattern_image_data

async def generate_hairline_with_mask_enhanced(input_image_data: bytes, input_mime_type: str,
                                    mask_image_data: bytes, timeframe: str, 
                                    hair_density: float, hair_type: str, 
                                    hair_color: str, pattern_type: str, request_id: str = ""):
    """IMPROVED: More reliable mask-based generation with better parameter enforcement"""
    logger.info(f"MASK-GEN-{request_id}: Starting IMPROVED mask-based generation")
    
    # Validate inputs
    if hair_density <= 0 or hair_density > 1:
        logger.warning(f"MASK-GEN-{request_id}: Invalid density {hair_density}, clamping")
        hair_density = max(0.1, min(1.0, hair_density))
    
    if not mask_image_data or len(mask_image_data) < 100:
        logger.error(f"MASK-GEN-{request_id}: Invalid mask data")
        return None
    
    # Enhanced logging
    logger.info(f"MASK-GEN-{request_id}: Generation parameters:")
    logger.info(f"  - Hair Color: {hair_color}")
    logger.info(f"  - Hair Type: {hair_type}")
    logger.info(f"  - Hair Density: {hair_density * 100:.1f}%")
    logger.info(f"  - Pattern Type: {pattern_type}")
    logger.info(f"  - Timeframe: {timeframe}")
    logger.info(f"  - Input image: {len(input_image_data)/1024:.1f}KB")
    logger.info(f"  - Mask image: {len(mask_image_data)/1024:.1f}KB")
    
    try:
        # Build improved prompt
        prompt = build_mask_based_hairline_prompt(timeframe, hair_density, hair_type, hair_color, pattern_type, request_id)
        
        # IMPROVED: Enhanced mask preprocessing
        enhanced_mask_data = convert_pattern_to_white_mask(mask_image_data, request_id)
        
        # Validate mask conversion
        if len(enhanced_mask_data) < 100:
            logger.warning(f"MASK-GEN-{request_id}: Mask conversion failed, using original")
            enhanced_mask_data = mask_image_data
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
        
        # Log exact prompt for debugging
        logger.info(f"MASK-GEN-{request_id}: Exact prompt being sent:")
        logger.info(f"{'='*50}")
        logger.info(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        logger.info(f"{'='*50}")
        
        # IMPROVED: More reliable model selection and retry logic
        models_to_try = ["gemini-2.5-flash-image-preview"
        ]
        
        for model_name in models_to_try:
            max_retries = 2  # Fewer retries per model, but more models
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"MASK-GEN-{request_id}: {model_name} attempt {attempt + 1}/{max_retries}")
                    
                    model = genai.GenerativeModel(model_name)
                    
                    # IMPROVED: Consistent generation config
                    generation_config = {
                        "temperature": 0.3,  # Lower for consistency
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_output_tokens": 4096,
                    }
                    
                    # Generate with timeout
                    response = await asyncio.wait_for(
                        asyncio.to_thread(model.generate_content, content, generation_config=generation_config),
                        timeout=90  # Longer timeout for mask-based generation
                    )
                    
                    if not response:
                        logger.warning(f"MASK-GEN-{request_id}: Empty response")
                        continue
                    
                    # IMPROVED: Enhanced image extraction with validation
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
                    await asyncio.sleep(2)  # Brief pause before retry
                    
                except Exception as e:
                    logger.error(f"MASK-GEN-{request_id}: Attempt {attempt + 1} failed: {str(e)}")
                    if "quota" in str(e).lower() or "limit" in str(e).lower():
                        logger.error(f"MASK-GEN-{request_id}: Rate limit hit, trying next model")
                        break
                    if attempt == max_retries - 1:
                        break
                    await asyncio.sleep(1)
    
        logger.error(f"MASK-GEN-{request_id}: All generation attempts failed")
        return None
        
    except Exception as e:
        logger.error(f"MASK-GEN-{request_id}: Critical error: {str(e)}")
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
    """IMPROVED: Enhanced image extraction with better validation"""
    logger.info(f"EXTRACT-{request_id}: Enhanced image extraction")
    
    try:
        # Method 1: Direct parts access
        if hasattr(response, 'parts') and response.parts:
            for i, part in enumerate(response.parts):
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'data') and part.inline_data.data:
                        image_data = part.inline_data.data
                        
                        # Validate and decode
                        decoded_data = decode_and_validate_image(image_data, f"{request_id}-parts-{i}")
                        if decoded_data:
                            return decoded_data
        
        # Method 2: Candidates access
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
        
        logger.warning(f"EXTRACT-{request_id}: No valid image data found in response")
        return None
        
    except Exception as e:
        logger.error(f"EXTRACT-{request_id}: Error in enhanced extraction: {str(e)}")
        return None
    
# Face Detection Functions (keeping existing ones)
def detect_face_with_mediapipe(image_data: bytes, request_id: str = "") -> dict:
    """Detect face using MediaPipe and return detection results with landmarks"""
    logger.info(f"FACE-DETECT-{request_id}: Starting face detection with MediaPipe...")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"FACE-DETECT-{request_id}: Failed to decode image")
            return {
                "face_detected": False,
                "confidence": 0.0,
                "error": "Failed to decode image",
                "landmarks": None,
                "face_bounds": None
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
                logger.warning(f"FACE-DETECT-{request_id}: No face detected")
                return {
                    "face_detected": False,
                    "confidence": 0.0,
                    "error": "No face detected",
                    "landmarks": None,
                    "face_bounds": None
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
            
            # Create visualization
            vis_image = image.copy()
            cv2.rectangle(vis_image, 
                         (int(face_left), int(face_top)), 
                         (int(face_right), int(face_bottom)), 
                         (0, 255, 0), 2)
            
            for i, landmark in enumerate(landmarks):
                if i % 10 == 0:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)
            
            cv2.putText(vis_image, f"Face Detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection_filename = f"{timestamp}_{request_id}_face_detection.jpg"
            detection_path = os.path.join("detections", detection_filename)
            cv2.imwrite(detection_path, vis_image)
            
            logger.info(f"FACE-DETECT-{request_id}: Face detected successfully!")
            logger.info(f"  - Confidence: {confidence:.1%}")
            logger.info(f"  - Detection saved: {detection_filename}")
            
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
        logger.error(f"FACE-DETECT-{request_id}: Error in face detection: {str(e)}")
        return {
            "face_detected": False,
            "confidence": 0.0,
            "error": f"Detection error: {str(e)}",
            "landmarks": None,
            "face_bounds": None
        }
 
def generate_hairlines_and_scalp_regions(image_data: bytes, face_bounds: dict, request_id: str = "") -> dict:
    """FIXED: Generate different hairline patterns with FILLED GREEN REGIONS for better detection"""
    logger.info(f"HAIRLINE-GEN-{request_id}: Generating hairline patterns with FILLED regions...")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"HAIRLINE-GEN-{request_id}: Failed to decode image")
            return {"error": "Failed to decode image"}

        h, w, _ = image.shape
        
        face_left = face_bounds["left"]
        face_right = face_bounds["right"]
        face_top = face_bounds["top"]
        face_bottom = face_bounds["bottom"]
        face_width = face_bounds["width"]
        face_height = face_bounds["height"]
        center_x = int(face_bounds["center_x"])
        
        generated_patterns = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Hairline M Pattern (keep existing - works fine)
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

        # FIXED Crown - SOLID FILLED GREEN ELLIPSE for better detection
        img_crown = image.copy()
        crown_y = int(face_top - (face_width * 0.5))
        crown_center = (center_x, crown_y)
        axes_crown = (int(face_width * 0.35), int(face_width * 0.2))  # Slightly smaller for better targeting
        
        # FIXED: Fill the entire ellipse area with SOLID GREEN color
        cv2.ellipse(img_crown, crown_center, axes_crown, 0, 0, 360, (0, 255, 0), -1)  # -1 = FILLED
        
        # NO transparency blending - keep it solid green for better detection
        # Draw GREEN ellipse outline for extra definition
        cv2.ellipse(img_crown, crown_center, axes_crown, 0, 0, 360, (0, 255, 0), 4)  # Thicker outline
        
        crown_filename = f"{timestamp}_{request_id}_crown.jpg"
        crown_path = os.path.join("detections", crown_filename)
        cv2.imwrite(crown_path, img_crown)
        generated_patterns["crown"] = crown_path
        logger.info(f"  - Crown pattern: SOLID GREEN ellipse at {crown_center} with axes {axes_crown}")

        # FIXED Full Scalp - SOLID FILLED GREEN ELLIPSE 
        img_scalp = image.copy()
        scalp_center = (center_x, int(face_top - face_height * 0.6))
        axes_scalp = (int(face_width * 0.5), int(face_height * 0.7))  # Larger for full scalp
        
        # FIXED: Fill the entire ellipse area with SOLID GREEN color
        cv2.ellipse(img_scalp, scalp_center, axes_scalp, 0, 0, 360, (0, 255, 0), -1)  # -1 = FILLED
        
        # NO transparency blending - keep it solid green for better detection
        # Draw GREEN ellipse outline for extra definition  
        cv2.ellipse(img_scalp, scalp_center, axes_scalp, 0, 0, 360, (0, 255, 0), 4)  # Thicker outline
        
        scalp_filename = f"{timestamp}_{request_id}_scalp.jpg"
        scalp_path = os.path.join("detections", scalp_filename)
        cv2.imwrite(scalp_path, img_scalp)
        generated_patterns["full_scalp"] = scalp_path
        logger.info(f"  - Full scalp pattern: SOLID GREEN ellipse at {scalp_center} with axes {axes_scalp}")

        # FIXED Mid-Crown - SOLID FILLED GREEN ELLIPSE
        img_mid_crown = image.copy()
        mid_crown_center = (center_x, int(scalp_center[1] - axes_scalp[1] * 0.3))  # Position between crown and full scalp
        axes_mid = (int(face_width * 0.25), int(face_width * 0.12))  # Medium size
        
        # FIXED: Fill the entire ellipse area with SOLID GREEN color
        cv2.ellipse(img_mid_crown, mid_crown_center, axes_mid, 0, 0, 360, (0, 255, 0), -1)  # -1 = FILLED
        
        # NO transparency blending - keep it solid green for better detection
        # Draw GREEN ellipse outline for extra definition
        cv2.ellipse(img_mid_crown, mid_crown_center, axes_mid, 0, 0, 360, (0, 255, 0), 4)  # Thicker outline
        
        mid_crown_filename = f"{timestamp}_{request_id}_mid_crown.jpg"
        mid_crown_path = os.path.join("detections", mid_crown_filename)
        cv2.imwrite(mid_crown_path, img_mid_crown)
        generated_patterns["mid_crown"] = mid_crown_path
        logger.info(f"  - Mid-crown pattern: SOLID GREEN ellipse at {mid_crown_center} with axes {axes_mid}")

        logger.info(f"HAIRLINE-GEN-{request_id}: Generated {len(generated_patterns)} SOLID GREEN patterns for better mask detection")

        return {
            "success": True,
            "patterns": generated_patterns,
            "pattern_count": len(generated_patterns),
            "generation_method": "solid_filled_ellipses"
        }

    except Exception as e:
        logger.error(f"HAIRLINE-GEN-{request_id}: Error generating patterns: {str(e)}")
        return {"error": f"Pattern generation error: {str(e)}"}

async def generate_face_detection_based_pattern(input_data, content_type, saved_settings, pattern_type,
                                               density_3m, density_8m, hair_type, hair_color, timeframe, request_id):
    """Generate Crown/Mid Crown/Full Scalp using face detection patterns"""
    logger.info(f"FACE-PATTERN-{request_id}: Generating {pattern_type} using face detection")
    
    # Map pattern types to detection file patterns
    pattern_mapping = {
        "Crown": "crown",
        "Mid Crown": "mid_crown", 
        "Full Scalp": "full_scalp"
    }
    
    detection_pattern = pattern_mapping.get(pattern_type, "crown")
    
    # FIXED: Look for face detection pattern files
    detection_files = []
    if os.path.exists("detections"):
        detection_files = [f for f in os.listdir("detections") if detection_pattern in f.lower()]
    
    if not detection_files:
        raise HTTPException(status_code=404, detail=f"No {pattern_type} pattern found. Please run face detection first.")
    
    # Get the most recent detection file
    latest_detection = max(detection_files, key=lambda x: os.path.getmtime(os.path.join("detections", x)))
    detection_path = os.path.join("detections", latest_detection)
    
    logger.info(f"FACE-PATTERN-{request_id}: Using detection file: {latest_detection}")
    
    # Convert face detection pattern to proper mask
    with open(detection_path, "rb") as f:
        pattern_data = f.read()
    
    # Convert to FreeMark-style mask
    mask_data = convert_to_freemark_style_mask(pattern_data, pattern_type, request_id)
    
    # Generate both timeframes
    generated_3m = await generate_hairline_with_mask_enhanced(
        input_data, content_type, mask_data,
        "3months", density_3m, hair_type, hair_color, pattern_type, f"{request_id}-3m"
    )
    
    generated_8m = await generate_hairline_with_mask_enhanced(
        input_data, content_type, mask_data,
        "8months", density_8m, hair_type, hair_color, pattern_type, f"{request_id}-8m"
    )
    
    if not generated_3m or not generated_8m:
        raise HTTPException(status_code=500, detail=f"Failed to generate {pattern_type} images")
    
    return {
        "request_id": request_id,
        "image": base64.b64encode(generated_3m if timeframe == "3months" else generated_8m).decode("utf-8"),
        "image_3months": base64.b64encode(generated_3m).decode("utf-8"),
        "image_8months": base64.b64encode(generated_8m).decode("utf-8"),
        "has_both_timeframes": True,
        "generation_mode": f"face_detection_{pattern_type.lower().replace(' ', '_')}",
        "pattern_type": pattern_type
    }

async def generate_custom_hairline_pattern(input_data, content_type, saved_settings,
                                         density_3m, density_8m, hair_type, hair_color, timeframe, request_id):
    """Generate M/Z/Curved hairline patterns"""
    saved_pattern = saved_settings.get("hairline_pattern", "M_pattern")
    logger.info(f"HAIRLINE-PATTERN-{request_id}: Generating {saved_pattern} hairline")
    
    mask_paths = saved_settings.get("mask_paths", {})
    
    # Check if we have a saved hairline mask
    if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
        logger.info(f"HAIRLINE-PATTERN-{request_id}: Using saved mask: {mask_paths['hairline_mask']}")
        with open(mask_paths["hairline_mask"], "rb") as f:
            mask_data = f.read()
        
        # Generate both timeframes using mask
        generated_3m = await generate_hairline_with_mask_enhanced(
            input_data, content_type, mask_data,
            "3months", density_3m, hair_type, hair_color, saved_pattern, f"{request_id}-3m"
        )
        
        generated_8m = await generate_hairline_with_mask_enhanced(
            input_data, content_type, mask_data,
            "8months", density_8m, hair_type, hair_color, saved_pattern, f"{request_id}-8m"
        )
    else:
        logger.info(f"HAIRLINE-PATTERN-{request_id}: Using coordinate-based generation for {saved_pattern}")
        # Fallback to coordinate-based generation
        image_width = saved_settings.get("image_dimensions", {}).get("width", 800)
        image_height = saved_settings.get("image_dimensions", {}).get("height", 600)
        
        coordinates_data = generate_hairline_coordinates_from_face_detection(
            image_width, image_height, request_id, saved_pattern
        )
        
        generated_3m = await generate_freemark_hair(
            input_data, content_type, coordinates_data,
            "3months", density_3m, hair_type, hair_color, f"{request_id}-3m"
        )
        
        generated_8m = await generate_freemark_hair(
            input_data, content_type, coordinates_data,
            "8months", density_8m, hair_type, hair_color, f"{request_id}-8m"
        )
    
    return {
        "request_id": request_id,
        "image": base64.b64encode(generated_3m if timeframe == "3months" else generated_8m).decode("utf-8"),
        "image_3months": base64.b64encode(generated_3m).decode("utf-8"),
        "image_8months": base64.b64encode(generated_8m).decode("utf-8"),
        "has_both_timeframes": True,
        "generation_mode": f"hairline_{saved_pattern}",
        "pattern_type": saved_pattern
    }

async def generate_freemark_mode(input_data, content_type, saved_settings,
                               density_3m, density_8m, hair_type, hair_color, timeframe, request_id):
    """Generate using user-drawn FreeMark masks"""
    logger.info(f"FREEMARK-{request_id}: Using FreeMark mode")
    
    mask_paths = saved_settings.get("mask_paths", {})
    
    # Check for required masks
    missing = []
    if "mask_3months" not in mask_paths: missing.append("3months")
    if "mask_8months" not in mask_paths: missing.append("8months")
    
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing FreeMark masks: {', '.join(missing)}")
    
    # Generate both timeframes
    with open(mask_paths["mask_3months"], "rb") as f:
        mask_3m_data = f.read()
    with open(mask_paths["mask_8months"], "rb") as f:
        mask_8m_data = f.read()
    
    generated_3m = await generate_hairline_with_mask_enhanced(
        input_data, content_type, mask_3m_data,
        "3months", density_3m, hair_type, hair_color, "FreeMark", f"{request_id}-3m"
    )
    
    generated_8m = await generate_hairline_with_mask_enhanced(
        input_data, content_type, mask_8m_data,
        "8months", density_8m, hair_type, hair_color, "FreeMark", f"{request_id}-8m"
    )
    
    return {
        "request_id": request_id,
        "image": base64.b64encode(generated_3m if timeframe == "3months" else generated_8m).decode("utf-8"),
        "image_3months": base64.b64encode(generated_3m).decode("utf-8"),
        "image_8months": base64.b64encode(generated_8m).decode("utf-8"),
        "has_both_timeframes": True,
        "generation_mode": "freemark_dual_masks",
        "pattern_type": "FreeMark"
    }

# Face Detection Endpoint
@app.post("/detect-face")
async def detect_face_endpoint(image: UploadFile = File(...)):
    """Detect face in uploaded image and return detection results"""
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"DETECT-{request_id}: NEW FACE DETECTION REQUEST")
    
    if not image.content_type or not image.content_type.startswith("image/"):
        logger.error(f"DETECT-{request_id}: Invalid file type: {image.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        logger.info(f"INPUT-{request_id}: Reading uploaded image...")
        image_data = await image.read()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{request_id}_upload_{image.filename}"
        upload_path = os.path.join("uploads", upload_filename)
        
        with open(upload_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"INPUT-{request_id}: Saved upload: {upload_filename} ({len(image_data) / 1024:.1f} KB)")
        
        detection_result = detect_face_with_mediapipe(image_data, request_id)
        
        pattern_result = None
        if detection_result["face_detected"]:
            pattern_result = generate_hairlines_and_scalp_regions(
                image_data, detection_result["face_bounds"], request_id
            )
            
            
            
        response_data = {
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
        
        if detection_result["face_detected"]:
            logger.info(f"DETECT-{request_id}: SUCCESS - Face detected with {detection_result['confidence']:.1%} confidence")
            if pattern_result and pattern_result.get("success"):
                logger.info(f"  - Generated {pattern_result['pattern_count']} hairline patterns")
        else:
            logger.info(f"DETECT-{request_id}: No face detected - FreeMark only mode")
        
        return response_data
        
    except Exception as e:
        logger.error(f"DETECT-{request_id}: CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

# Load prompts configuration
def load_prompts_config():
    """Load prompts configuration from prompt.json file"""
    try:
        with open("prompt.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info("PROMPTS: Configuration loaded successfully")
            return config
    except FileNotFoundError:
        logger.warning("WARNING: prompt.json file not found! Using default configuration")
        return {
            "generation_prompts": {
                "freemark_generation": {
                    "base_instruction": "Generate realistic hair growth simulation",
                    "timeframe_specs": {
                        "3months": {"prompt_addition": "Show early hair growth"},
                        "8months": {"prompt_addition": "Show mature hair growth"}
                    },
                    "hair_types": {"Straight Hair": "natural straight hair"},
                    "color_instructions": {"#000000": "black hair"},
                    "final_reminders": "Make it look natural and realistic."
                },
                "standard_generation": {
                    "base_template": "Generate hair growth for {region} area",
                    "region_focus": {
                        "Hairline": {"description": "Focus on hairline area"},
                        "Crown": {"description": "Focus on crown area"}
                    },
                    "timeframe_specs": {
                        "3months": {"description": "early growth"},
                        "8months": {"description": "mature growth"}
                    }
                }
            }
        }
    except json.JSONDecodeError as e:
        logger.error(f"ERROR: Invalid JSON in prompt.json: {e}")
        raise HTTPException(status_code=500, detail="Invalid prompts configuration")

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
    # Pattern/mask files
    mask_3months: Optional[UploadFile] = File(None),
    mask_8months: Optional[UploadFile] = File(None),
    hairline_mask: Optional[UploadFile] = File(None),  # NEW: Hairline pattern mask
    # Hairline pattern data
    hairline_pattern: str = Form("M_pattern"),
    hairline_points: Optional[str] = Form(None)
):
    """Save user settings and pattern/mask data for later generation - NO GENERATION HERE"""
    request_id = generate_request_id()
    logger.info(f"SAVE-SETTINGS-{request_id}: Saving user settings and pattern data (NO GENERATION)")
    
    # Log all received settings
    logger.info(f"SAVE-SETTINGS-{request_id}: Received settings from frontend:")
    logger.info(f"  - Hair Color: {hair_color}")
    logger.info(f"  - Hair Type: {hair_type}")
    logger.info(f"  - Hair Line Type: {hair_line_type}")
    logger.info(f"  - Hair Density 3M: {hair_density_3m}")
    logger.info(f"  - Hair Density 8M: {hair_density_8m}")
    logger.info(f"  - Timeframe: {timeframe}")
    logger.info(f"  - Face Detected: {face_detected}")
    logger.info(f"  - Hairline Pattern: {hairline_pattern}")
    logger.info(f"  - Hairline Mask: {'Yes' if hairline_mask else 'No'}")
    logger.info(f"  - 3-Month Mask: {'Yes' if mask_3months else 'No'}")
    logger.info(f"  - 8-Month Mask: {'Yes' if mask_8months else 'No'}")
    
    # ENHANCED: Log mask file details
    if mask_3months:
        logger.info(f"  - 3-Month Mask Details: filename={mask_3months.filename}, content_type={mask_3months.content_type}")
    if mask_8months:
        logger.info(f"  - 8-Month Mask Details: filename={mask_8months.filename}, content_type={mask_8months.content_type}")
    if hairline_mask:
        logger.info(f"  - Hairline Mask Details: filename={hairline_mask.filename}, content_type={hairline_mask.content_type}")
    
    try:
        logger.info(f"INPUT-{request_id}: Reading uploaded image...")
        image_data = await image.read()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_filename = f"{timestamp}_{request_id}_upload_{image.filename}"
        upload_path = os.path.join("uploads", upload_filename)
        
        with open(upload_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"INPUT-{request_id}: Saved upload: {upload_filename} ({len(image_data) / 1024:.1f} KB)")
        
        input_image = Image.open(io.BytesIO(image_data))
        image_width, image_height = input_image.size
        
        # Create settings directory if it doesn't exist
        settings_dir = os.path.join("logs", "saved_settings")
        os.makedirs(settings_dir, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(settings_dir, f"{request_id}_original.png")
        input_image.save(original_path)
        
        # Save settings data
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
        
        # Save hairline points if provided
        if hairline_points:
            try:
                points_data = json.loads(hairline_points)
                settings_data["hairline_points"] = points_data
                logger.info(f"  - Saved hairline points: {len(points_data.get('inner', []))} inner, {len(points_data.get('outer', []))} outer")
            except json.JSONDecodeError:
                logger.warning(f"  - Invalid hairline points JSON, skipping")
        
        # Save masks if provided - NO GENERATION, ONLY SAVING
        mask_paths = {}
        
        # Save FreeMark masks
        if mask_3months:
            mask_3m_data = await mask_3months.read()
            mask_3m_path = os.path.join(settings_dir, f"{request_id}_mask_3months.png")
            with open(mask_3m_path, "wb") as f:
                f.write(mask_3m_data)
            mask_paths["mask_3months"] = mask_3m_path
            logger.info(f"  - Saved 3-month mask: {len(mask_3m_data)} bytes")
            
        if mask_8months:
            mask_8m_data = await mask_8months.read()
            mask_8m_path = os.path.join(settings_dir, f"{request_id}_mask_8months.png")
            with open(mask_8m_path, "wb") as f:
                f.write(mask_8m_data)
            mask_paths["mask_8months"] = mask_8m_path
            logger.info(f"  - Saved 8-month mask: {len(mask_8m_data)} bytes")
            logger.info(f"  - 8-month mask saved to: {mask_8m_path}")
        else:
            logger.warning(f"SAVE-SETTINGS-{request_id}: ⚠️  NO 8-MONTH MASK provided by frontend!")
        
        # Save hairline pattern mask if provided - Convert to FreeMark style (black outer, white inner)
        if hairline_mask:
            hairline_mask_data = await hairline_mask.read()
            
            # Convert to FreeMark style mask (black background, white pattern areas)
            # Pass the pattern type for specific processing
            freemark_style_mask = convert_to_freemark_style_mask(hairline_mask_data, request_id, hairline_pattern)
            
            hairline_mask_path = os.path.join(settings_dir, f"{request_id}_hairline_mask.png")
            
            with open(hairline_mask_path, "wb") as f:
                f.write(freemark_style_mask)
            
            mask_paths["hairline_mask"] = hairline_mask_path
            logger.info(f"  - Saved {hairline_pattern} mask (FreeMark style): {len(freemark_style_mask)} bytes")
        
        settings_data["mask_paths"] = mask_paths
        
        # Save settings JSON
        settings_path = os.path.join(settings_dir, f"{request_id}_settings.json")
        with open(settings_path, "w") as f:
            json.dump(settings_data, f, indent=2)
        
        logger.info(f"SAVE-SETTINGS-{request_id}: Settings saved successfully (NO GENERATION PERFORMED)")
        logger.info(f"  - Original image: {original_path}")
        logger.info(f"  - Settings file: {settings_path}")
        logger.info(f"  - Masks saved: {len(mask_paths)}")
        
        return {
            "success": True,
            "request_id": request_id,
            "message": "Settings and pattern data saved successfully - use generate-with-saved-pattern to create hair images",
            "saved_files": {
                "original_image": original_path,
                "settings": settings_path,
                "masks": mask_paths
            }
        }
        
    except Exception as e:
        logger.error(f"SAVE-SETTINGS-{request_id}: Error saving settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")

# Utility Functions
def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

def generate_hairline_coordinates_from_face_detection(image_width: int, image_height: int, request_id: str = "", pattern_type: str = "M_pattern") -> dict:
    """Generate hairline coordinates based on image dimensions and pattern type"""
    logger.info(f"HAIRLINE-COORDS-{request_id}: Generating {pattern_type} coordinates from image dimensions")
    
    try:
        # Simulate face bounds based on image dimensions (typical face positioning)
        face_width = int(image_width * 0.4)  # Face is typically 40% of image width
        face_height = int(image_height * 0.5)  # Face is typically 50% of image height
        face_left = int(image_width * 0.3)  # Face starts at 30% from left
        face_top = int(image_height * 0.2)   # Face starts at 20% from top
        face_right = face_left + face_width
        center_x = face_left + face_width // 2
        
        logger.info(f"  - Simulated face bounds: {face_left},{face_top} to {face_right},{face_top + face_height}")
        
        # Generate different hairline patterns based on type
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
                [face_left + face_width * 0.1, face_top - 35],   # Top-left
                [face_right - face_width * 0.1, face_top - 35],  # Top-right
                [face_left + face_width * 0.15, face_top - 10],  # Bottom-left
                [face_right - face_width * 0.1, face_top - 5]    # Bottom-right
            ]
        elif pattern_type == "Curved_pattern":
            hairline_points = [
                [face_left + face_width * 0.1, face_top - 20],
                [face_left + face_width * 0.3, face_top - 15],
                [center_x, face_top - 10],
                [face_right - face_width * 0.3, face_top - 15],
                [face_right - face_width * 0.1, face_top - 20]
            ]
        else:
            # Default to M-pattern
            hairline_points = [
                [face_left + face_width * 0.1, face_top - 20],
                [face_left + face_width * 0.25, face_top - 40], 
                [center_x, face_top - 10],
                [face_right - face_width * 0.25, face_top - 40],
                [face_right - face_width * 0.1, face_top - 20]
            ]
        
        # ENHANCED: Convert to coordinate regions format with half-bald optimization
        regions = []
        total_pixels = image_width * image_height
        
        for i, point in enumerate(hairline_points):
            x, y = point
            # ENHANCED: Larger regions for better half-bald coverage
            region_size = 35 if pattern_type == "Z_pattern" else 30  # Increased from 25/20
            
            # ENHANCED: Calculate actual area for priority assignment
            region_width = min(region_size * 2, image_width - max(0, x - region_size))
            region_height = min(region_size * 2, image_height - max(0, y - region_size))
            actual_area = region_width * region_height
            area_ratio = actual_area / total_pixels
            
            # ENHANCED: Priority-based region handling for half-bald optimization
            if area_ratio >= 0.05:  # Ultra-high priority (5%+ of image)
                priority = "ultra_high"
                density_multiplier = 1.8
            elif area_ratio >= 0.02 or (i in [1, 3] and pattern_type == "M_pattern"):  # High priority key points
                priority = "high" 
                density_multiplier = 1.5
            elif area_ratio >= 0.01:  # Medium priority
                priority = "medium"
                density_multiplier = 1.2
            else:  # Standard priority
                priority = "standard"
                density_multiplier = 1.0
            
            # ENHANCED: Create region with better coordinate detection for sparse hair
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
                "region_ratio": actual_area / (len(hairline_points) * region_size * region_size * 4),
                "density_priority": priority,
                "density_multiplier": density_multiplier,
                "growth_direction": "forward_natural" if y < image_height * 0.3 else "natural_backward",
                "region_id": f"hairline_point_{i+1}",
                "half_bald_optimized": True
            }
            regions.append(region)
            
        # ENHANCED: Sort regions by priority for processing order
        priority_order = {"ultra_high": 4, "high": 3, "medium": 2, "standard": 1}
        regions.sort(key=lambda r: priority_order.get(r["density_priority"], 0), reverse=True)
        
        # ENHANCED: Log priority distribution for half-bald optimization
        priority_distribution = {
            priority: len([r for r in regions if r["density_priority"] == priority])
            for priority in ["ultra_high", "high", "medium", "standard"]
        }
        
        logger.info(f"  - Generated {len(regions)} {pattern_type} coordinate regions with half-bald optimization")
        logger.info(f"  - Priority distribution: {priority_distribution}")
        for i, region in enumerate(regions):
            logger.info(f"    ✓ Region {i+1}: Priority={region['density_priority'].upper()}, Multiplier={region['density_multiplier']}x, Area={region['area']} ({region['area_ratio']:.3f}%)")
        
        return {
            "regions": regions,
            "total_regions": len(regions),
            "generation_method": "enhanced_hairline_face_detection",
            "hairline_pattern": pattern_type,
            "hairline_points": hairline_points,
            "detection_method": "enhanced_half_bald_optimized",
            "total_marked_area": sum(r["area"] for r in regions),
            "coverage_ratio": sum(r["area"] for r in regions) / total_pixels,
            "priority_distribution": priority_distribution
        }
        
    except Exception as e:
        logger.error(f"HAIRLINE-COORDS-{request_id}: Error generating coordinates: {str(e)}")
        return {"error": f"Failed to generate hairline coordinates: {str(e)}"}

if __name__ == "__main__":
    logger.info("STARTUP: Starting Hair Growth API - Consolidated Individual Endpoint")
    uvicorn.run(app, host="0.0.0.0", port=8000)
