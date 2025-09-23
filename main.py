# Updated main.py with consolidated /individual endpoint
print("Starting main.py...")

import sys
import os
import logging
import traceback
from datetime import datetime
import json

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

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'google.generativeai',
        'cv2', 'mediapipe', 'PIL', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'google.generativeai':
                import google.generativeai as genai
            else:
                __import__(package)
        except ImportError as e:
            missing_packages.append(package)
            logger.error(f"Missing package: {package} - {e}")
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed")
    return True

try:
    
    GEMINI_API_KEY = "AIzaSyARkGwZERa_Gzy_XwlyjcBw7-U02o4YKDg"  # Replace with your valid API key 
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY is required. Please set it in main.py.")
        raise ValueError("GEMINI_API_KEY is required. Please set it in main.py.")

    # Import dependencies
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import base64
    import google.generativeai as genai
    from google.generativeai.types import BlobDict
    from typing import Optional
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    import io
    import uuid
    import asyncio
    import mediapipe as mp
    import os
    import io
    import cv2
    import json
    import base64
    import asyncio
    import logging
    import numpy as np
    from PIL import Image, ImageDraw
    from typing import Optional, Dict, Any
    from datetime import datetime
    import mediapipe as mp

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

# ---------------------------
# FIXED: Coordinate-Based Hair Growth Functions
# ---------------------------

def extract_coordinates_from_markings(marking_image_data: bytes, request_id: str = "") -> dict:
    """ENHANCED: Improved coordinate extraction for half-bald detection with priority-based regions"""
    logger.info(f"COORDS-{request_id}: Starting enhanced coordinate extraction for half-bald optimization...")
    
    try:
        # Load image
        nparr = np.frombuffer(marking_image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Could not decode marking image"}
        
        height, width = image.shape[:2]
        total_pixels = width * height
        logger.info(f"COORDS-{request_id}: Image size {width}x{height}")
        
        # ENHANCED: Multi-method detection for better half-bald region detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Adaptive thresholding for varied lighting
        adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Method 2: Multiple fixed thresholds
        thresholds = [200, 180, 160, 140, 120, 100, 80]  # Extended range for sparse hair
        threshold_masks = []
        
        for threshold in thresholds:
            _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            _, light_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            dark_coverage = cv2.countNonZero(dark_mask) / total_pixels
            light_coverage = cv2.countNonZero(light_mask) / total_pixels
            
            # Accept masks with reasonable coverage for half-bald detection
            if 0.005 < dark_coverage < 0.6:  # Lowered minimum for sparse markings
                threshold_masks.append((dark_mask, dark_coverage, f"dark_t{threshold}"))
            if 0.005 < light_coverage < 0.6:
                threshold_masks.append((light_mask, light_coverage, f"light_t{threshold}"))
        
        # Method 3: Edge detection for faint markings
        edges = cv2.Canny(gray, 50, 150)
        edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_coverage = cv2.countNonZero(edge_dilated) / total_pixels
        if 0.01 < edge_coverage < 0.4:
            threshold_masks.append((edge_dilated, edge_coverage, "edge_detection"))
        
        # Add adaptive mask
        adaptive_coverage = cv2.countNonZero(adaptive_mask) / total_pixels
        if 0.005 < adaptive_coverage < 0.6:
            threshold_masks.append((adaptive_mask, adaptive_coverage, "adaptive"))
        
        if not threshold_masks:
            logger.warning(f"COORDS-{request_id}: No suitable markings found with enhanced detection")
            return {"error": "No clear markings detected - try using more contrasted markings or larger marked areas"}
        
        # Select best mask based on coverage and quality
        best_mask, best_coverage, best_method = max(threshold_masks, key=lambda x: x[1] if x[1] < 0.3 else 0.3 - (x[1] - 0.3))
        logger.info(f"COORDS-{request_id}: Selected {best_method} mask with {best_coverage:.3f} coverage")
        
        # Enhanced cleanup for sparse regions
        kernel_close = np.ones((7, 7), np.uint8)  # Larger kernel for connecting sparse regions
        kernel_open = np.ones((3, 3), np.uint8)
        
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel_close)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours with hierarchy for better region detection
        contours, hierarchy = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"error": "No marking regions found after enhanced processing"}
        
        logger.info(f"COORDS-{request_id}: Found {len(contours)} contour regions")
        
        # ENHANCED: Process contours with priority-based classification
        regions = []
        min_area = max(50, total_pixels * 0.0005)  # Even smaller minimum for sparse hair
        max_area = total_pixels * 0.8
        
        # Calculate area ratios for priority assignment
        contour_areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) >= min_area]
        if not contour_areas:
            return {"error": "No valid regions after enhanced filtering"}
        
        total_marked_area = sum(contour_areas)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < min_area:
                logger.info(f"  Region {i+1}: REJECTED (too small: {area:.0f} < {min_area:.0f})")
                continue
            if area > max_area:
                logger.info(f"  Region {i+1}: REJECTED (too large: {area:.0f} > {max_area:.0f})")
                continue
            
            # Enhanced bounding rectangle with adaptive padding
            x, y, w, h = cv2.boundingRect(contour)
            padding = max(3, min(10, int(min(w, h) * 0.1)))  # Adaptive padding
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(width - x_pad, w + 2*padding)
            h_pad = min(height - y_pad, h + 2*padding)
            
            # Calculate center with moment-based precision
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                center_x = x + w // 2
                center_y = y + h // 2
            
            # ENHANCED: Priority-based classification for half-bald optimization
            area_ratio = area / total_pixels
            region_ratio = area / total_marked_area if total_marked_area > 0 else 0
            
            # Assign priority based on prompts.json thresholds
            if area_ratio >= 0.05:  # Ultra-high priority (5%+ of image)
                priority = "ultra_high"
                density_multiplier = 1.8
            elif area_ratio >= 0.02:  # High priority (2%+ of image)
                priority = "high" 
                density_multiplier = 1.5
            elif area_ratio >= 0.01:  # Medium priority (1%+ of image)
                priority = "medium"
                density_multiplier = 1.2
            else:  # Standard priority
                priority = "standard"
                density_multiplier = 1.0
            
            # Determine growth direction based on location
            if center_y < height * 0.3:  # Upper region - likely hairline
                growth_direction = "forward_natural"
            elif center_y < height * 0.6:  # Middle region - crown area
                growth_direction = "radial_crown"
            else:  # Lower region
                growth_direction = "natural_backward"
            
            region = {
                "bbox": [x_pad, y_pad, w_pad, h_pad],
                "center": [center_x, center_y],
                "area": int(area),
                "area_ratio": area_ratio,
                "region_ratio": region_ratio,
                "density_priority": priority,
                "density_multiplier": density_multiplier,
                "growth_direction": growth_direction,
                "region_id": f"region_{i+1}",
                "half_bald_optimized": True
            }
            
            regions.append(region)
            logger.info(f"  ✓ Region {i+1}: Area={area:.0f} ({area_ratio:.3f}%), Priority={priority.upper()}, Multiplier={density_multiplier}x")
        
        if not regions:
            return {"error": "No valid regions after enhanced filtering"}
        
        # Sort regions by priority for processing order
        priority_order = {"ultra_high": 4, "high": 3, "medium": 2, "standard": 1}
        regions.sort(key=lambda r: priority_order.get(r["density_priority"], 0), reverse=True)
        
        logger.info(f"COORDS-{request_id}: ✓ Extracted {len(regions)} enhanced regions with half-bald optimization")
        
        return {
            "regions": regions,
            "image_dimensions": {"width": width, "height": height},
            "total_regions": len(regions),
            "detection_method": "enhanced_half_bald_optimized",
            "total_marked_area": int(total_marked_area),
            "coverage_ratio": total_marked_area / total_pixels,
            "priority_distribution": {
                priority: len([r for r in regions if r["density_priority"] == priority])
                for priority in ["ultra_high", "high", "medium", "standard"]
            }
        }
        
    except Exception as e:
        logger.error(f"COORDS-{request_id}: Enhanced extraction error: {str(e)}")
        return {"error": f"Enhanced coordinate extraction failed: {str(e)}"}

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
    """ENHANCED: Build FreeMark prompt using prompts.json configuration for half-bald optimization"""
    logger.info(f"PROMPT-{request_id}: Building enhanced FreeMark prompt for {timeframe} with half-bald optimization")
    
    # Load prompts configuration
    global PROMPTS_CONFIG
    if not PROMPTS_CONFIG:
        PROMPTS_CONFIG = load_prompts_config()
    
    freemark_config = PROMPTS_CONFIG.get("generation_prompts", {}).get("freemark_generation", {})
    
    # Get coordinate instructions with enhanced formatting
    coord_prompt = format_coordinates_for_prompt(coordinates_data, request_id)
    
    # Enhanced timeframe-specific instructions from prompts.json
    timeframe_specs = freemark_config.get("timeframe_specs", {})
    if timeframe in ["3months", "3 months", "3 Months"]:
        timeframe_key = "3months"
        base_multiplier = 1.4  # From prompts.json
        timeframe_desc = f"ENHANCED early regrowth with {hair_density:.1%} base density + {base_multiplier}x visibility multiplier"
        growth_type = "clearly visible emerging hair strands with enhanced thickness and contrast"
        visibility_req = "obvious_improvement"
    else:
        timeframe_key = "8months"
        base_multiplier = 1.6  # From prompts.json
        timeframe_desc = f"MAXIMUM mature growth with {hair_density:.1%} base density + {base_multiplier}x visibility multiplier"
        growth_type = "dramatically thick, full, established hair with maximum visual impact"
        visibility_req = "dramatic_transformation"
    
    # Enhanced hair color instructions from prompts.json
    color_instructions = freemark_config.get("color_instructions", {})
    if hair_color in color_instructions:
        color_desc = color_instructions[hair_color]
    elif hair_color == "#000000":
        color_desc = "DRAMATIC jet black with enhanced natural highlights, optimal depth, and maximum scalp contrast"
    else:
        color_desc = f"enhanced {hair_color} with dramatic shine, depth, and optimal contrast for maximum visibility"
    
    # Enhanced hair type descriptions from prompts.json
    hair_types = freemark_config.get("hair_types", {})
    if hair_type in hair_types:
        hair_type_desc = hair_types[hair_type]
    else:
        hair_type_desc = f"Enhanced {hair_type.lower()} with increased thickness, natural shine, and optimal contrast against scalp"
    
    # Get system role and base instructions from prompts.json
    system_role = freemark_config.get("system_role", "Expert medical illustration specialist")
    base_instruction = freemark_config.get("base_instruction", "")
    
    # Build the enhanced prompt using prompts.json structure
    prompt = f"""{system_role}

TASK: {freemark_config.get('task', 'Generate DRAMATICALLY VISIBLE hair regrowth simulation')}

{coord_prompt}

ENHANCED TIMEFRAME SPECIFICATION:
{timeframe_desc}
Growth characteristics: {growth_type}
Visibility requirement: {visibility_req}

ENHANCED GENERATION REQUIREMENTS:
- Hair type: {hair_type_desc}
- Hair color: {color_desc}
- Base density: {hair_density:.1%} + {base_multiplier}x enhancement multiplier
- Apply region-specific priority multipliers as specified in coordinate data
- Growth pattern: Natural direction with enhanced visibility optimization
- Preserve all facial features and skin texture from input image
- Create seamless integration with existing scalp/hair texture

CRITICAL RULES FROM ENHANCED CONFIGURATION:
"""
    
    # Add critical rules from prompts.json
    critical_rules = freemark_config.get("critical_rules", [])
    for rule in critical_rules:
        prompt += f"\n• {rule}"
    
    # Add timeframe-specific prompt addition
    if timeframe_key in timeframe_specs:
        timeframe_addition = timeframe_specs[timeframe_key].get("prompt_addition", "")
        if timeframe_addition:
            # Format the prompt addition with current values
            formatted_addition = timeframe_addition.format(
                hair_density=hair_density,
                timeframe=timeframe
            )
            prompt += f"\n\nTIMEFRAME-SPECIFIC ENHANCEMENT:\n{formatted_addition}"
    
    # Add final reminders from prompts.json
    final_reminders = freemark_config.get("final_reminders", "")
    if final_reminders:
        formatted_reminders = final_reminders.format(timeframe=timeframe)
        prompt += f"\n\n{formatted_reminders}"
    
    logger.info(f"PROMPT-{request_id}: Generated enhanced {len(prompt)} character prompt with half-bald optimization")
    return prompt

async def generate_freemark_hair(input_image_data: bytes, input_mime_type: str,
                               coordinates_data: dict, timeframe: str, 
                               hair_density: float, hair_type: str, 
                               hair_color: str, request_id: str = ""):
    """ENHANCED: FreeMark generation with fixed image extraction logic"""
    logger.info(f"GEN-{request_id}: Starting enhanced FreeMark generation")
    logger.info(f"  - Regions: {len(coordinates_data.get('regions', []))}")
    logger.info(f"  - Timeframe: {timeframe}")
    logger.info(f"  - Density: {hair_density * 100:.1f}%")
    
    # Build enhanced prompt using prompts.json configuration
    prompt = build_freemark_generation_prompt(coordinates_data, timeframe, hair_density, hair_type, hair_color, request_id)
    
    # Create content for Gemini API - FIXED FORMAT
    import base64
    
    # Convert bytes to base64 string for Gemini API
    image_b64 = base64.b64encode(input_image_data).decode('utf-8')
    
    # FIXED: Use proper Gemini API content structure
    content = [
        {
            "inline_data": {
                "mime_type": input_mime_type,
                "data": image_b64
            }
        },
        prompt
    ]
    
    # Try generation with multiple models and retries
    max_retries = 3
    models_to_try = ["gemini-2.5-flash-image-preview"]
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                logger.info(f"GEN-{request_id}: Attempt {attempt + 1}/{max_retries} - Sending to {model_name}...")
                
                model = genai.GenerativeModel(model_name)
                
                # FIXED: Use the correct API call method
                response = await asyncio.to_thread(model.generate_content, content)
                
                logger.info(f"GEN-{request_id}: Response type: {type(response).__name__}")
                
                # FIXED: Proper response parsing for image generation
                if hasattr(response, 'parts') and response.parts:
                    logger.info(f"GEN-{request_id}: Parts count: {len(response.parts)}")
                    
                    for i, part in enumerate(response.parts):
                        logger.info(f"GEN-{request_id}: Part {i}: {type(part).__name__}")
                        
                        # FIXED: Check for inline_data properly
                        if hasattr(part, 'inline_data') and part.inline_data:
                            inline_data = part.inline_data
                            logger.info(f"GEN-{request_id}: Found inline_data: {type(inline_data).__name__}")
                            
                            # FIXED: Handle different data formats
                            if hasattr(inline_data, 'data') and inline_data.data:
                                image_data = inline_data.data
                                logger.info(f"GEN-{request_id}: Found image data: {type(image_data).__name__}, length: {len(str(image_data))}")
                                
                                # FIXED: Handle base64 string vs bytes
                                if isinstance(image_data, str):
                                    # Data is base64 encoded string, decode it
                                    try:
                                        decoded_data = base64.b64decode(image_data)
                                        logger.info(f"GEN-{request_id}: ✓ Generated {len(decoded_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                        return decoded_data
                                    except Exception as decode_error:
                                        logger.error(f"GEN-{request_id}: Base64 decode error: {decode_error}")
                                        continue
                                elif isinstance(image_data, bytes):
                                    # Data is already bytes
                                    logger.info(f"GEN-{request_id}: ✓ Generated {len(image_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                    return image_data
                                else:
                                    logger.warning(f"GEN-{request_id}: Unexpected data type: {type(image_data)}")
                        
                        # FIXED: Alternative check for direct data attribute
                        elif hasattr(part, 'data') and part.data:
                            image_data = part.data
                            if isinstance(image_data, bytes):
                                logger.info(f"GEN-{request_id}: ✓ Generated {len(image_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                return image_data
                            elif isinstance(image_data, str):
                                try:
                                    decoded_data = base64.b64decode(image_data)
                                    logger.info(f"GEN-{request_id}: ✓ Generated {len(decoded_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                    return decoded_data
                                except Exception as decode_error:
                                    logger.error(f"GEN-{request_id}: Base64 decode error: {decode_error}")
                                    continue
                    
                    logger.warning(f"GEN-{request_id}: No valid image data found in response parts on attempt {attempt + 1}")
                
                # FIXED: Check if response has candidates (alternative structure)
                elif hasattr(response, 'candidates') and response.candidates:
                    logger.info(f"GEN-{request_id}: Found candidates: {len(response.candidates)}")
                    
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                            image_data = part.inline_data.data
                                            if isinstance(image_data, str):
                                                try:
                                                    decoded_data = base64.b64decode(image_data)
                                                    logger.info(f"GEN-{request_id}: ✓ Generated {len(decoded_data)/1024:.1f}KB image from candidates")
                                                    return decoded_data
                                                except Exception as decode_error:
                                                    logger.error(f"GEN-{request_id}: Candidates decode error: {decode_error}")
                                                    continue
                                            elif isinstance(image_data, bytes):
                                                logger.info(f"GEN-{request_id}: ✓ Generated {len(image_data)/1024:.1f}KB image from candidates")
                                                return image_data
                
                # FIXED: Log the actual response structure for debugging
                logger.warning(f"GEN-{request_id}: Could not extract image from response structure")
                logger.warning(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                if hasattr(response, 'parts') and response.parts:
                    for i, part in enumerate(response.parts):
                        logger.warning(f"Part {i} attributes: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                        
            except Exception as e:
                logger.error(f"GEN-{request_id}: Attempt {attempt + 1} failed with {model_name}: {str(e)}")
                logger.error(f"GEN-{request_id}: Exception type: {type(e).__name__}")
                if "Could not convert" in str(e):
                    logger.error(f"GEN-{request_id}: This appears to be a successful generation but with extraction issues")
                if attempt == max_retries - 1:
                    logger.error(f"GEN-{request_id}: All attempts failed with {model_name}")
                else:
                    await asyncio.sleep(1)  # Brief delay before retry
    
    logger.error(f"GEN-{request_id}: All models and attempts failed")
    return None

def build_mask_based_hairline_prompt(timeframe: str, hair_density: float, hair_type: str, 
                                   hair_color: str, pattern_type: str, request_id: str = "",
                                   input_image_size: int = 0, mask_image_size: int = 0, 
                                   white_pixel_count: int = 0, total_pixels: int = 0) -> str:
    """ENHANCED: Build mask-based hairline prompt with detailed logging and pattern support"""
    logger.info(f"HAIRLINE-PROMPT-{request_id}: Building mask-based hairline prompt")
    logger.info(f"HAIRLINE-PROMPT-{request_id}: Settings - Timeframe: {timeframe}, Density: {hair_density:.2f}, Type: {hair_type}, Color: {hair_color}, Pattern: {pattern_type}")
    
    try:
        # Get hairline mask prompt from prompts.json
        hairline_config = PROMPTS_CONFIG["generation_prompts"]["standard_generation"]["region_focus"]["Hairline"]
        enhancement_multiplier = hairline_config["enhancement_multiplier"]
        
        # Get timeframe-specific configuration
        timeframe_config = PROMPTS_CONFIG["generation_prompts"]["standard_generation"]["timeframe_specs"][timeframe]
        timeframe_multiplier = timeframe_config["enhancement_multiplier"]
        
        # Calculate final enhancement multiplier
        final_multiplier = enhancement_multiplier * timeframe_multiplier
        
        # Add color-specific instructions
        color_instructions = PROMPTS_CONFIG["generation_prompts"]["freemark_generation"]["color_instructions"]
        color_instruction = color_instructions.get(hair_color, f"Natural {hair_color} hair color with enhanced visibility and optimal contrast")
        
        # Get specific hair characteristics based on type
        hair_texture_desc = {
            "Straight Hair": "straight, smooth strands that lie flat against the scalp",
            "Wavy Hair": "gentle waves with natural S-curve patterns and medium volume",
            "Curly Hair": "defined curls with spiral patterns and natural volume",
            "Coarse Hair": "thick, strong strands with robust texture",
            "Fine Hair": "delicate, thin strands with subtle texture"
        }.get(hair_type, "natural hair strands")
        
        # Get color-specific instructions
        color_desc = {
            "#000000": "deep black hair with natural shine and highlights",
            "#654321": "rich dark brown with warm undertones", 
            "#964B00": "medium brown with natural variation",
            "#808080": "silver-gray with natural dimension",
            "#6F4E37": "coffee brown with depth and warmth",
            "#F0E2B6": "light golden blonde with natural highlights"
        }.get(hair_color, f"natural {hair_color} colored hair")
        
        # Build enhanced hairline-specific prompt
        complete_prompt = f"""PROFESSIONAL HAIR RESTORATION - MASK-GUIDED HAIR GROWTH

TASK: Generate realistic hair growth in specific regions using mask guidance.

INPUT ANALYSIS:
- Image 1: Original photo showing person with hair loss/balding areas
  → Source: User uploaded image ({input_image_size} bytes)
  → Purpose: Base image for hair restoration
- Image 2: MASK IMAGE with WHITE regions indicating hair growth areas  
  → Source: Generated/Enhanced mask ({mask_image_size} bytes)
  → White pixels: {white_pixel_count} / {total_pixels} total ({white_pixel_count/total_pixels*100:.1f}% coverage)
  → Purpose: Strict boundary template for hair generation
- Pattern Type: {pattern_type} hairline restoration pattern
- Request ID: {request_id}

MASK INTERPRETATION (CRITICAL - MAXIMUM CONTRAST):
🎯 WHITE AREAS in mask = MAXIMUM HAIR GROWTH ZONES (generate DENSE, THICK hair here)
   → 100% AGGRESSIVE hair generation in white pixels
   → MAXIMUM density and thickness in white regions
   → DRAMATIC hair growth - make it VERY VISIBLE
   → WHITE = FULL HAIR RESTORATION TARGET AREAS

🎯 BLACK AREAS in mask = ZERO GROWTH ZONES (absolutely NO hair generation)
   → 0% hair generation in black pixels
   → COMPLETELY preserve original appearance
   → NO modifications, NO new hair, NO changes whatsoever
   → BLACK = STRICT NO-TOUCH ZONES

🎯 EXTREME CONTRAST RULE: Create MAXIMUM difference between white and black areas
🎯 WHITE areas should have DRAMATICALLY MORE hair than black areas
🎯 The mask is a STRICT BOUNDARY - respect it with 100% precision
🎯 NEVER blend or fade between white and black - maintain SHARP contrast

HAIR SPECIFICATIONS FOR GENERATION:
🎯 Hair Type: {hair_type}
   → Generate {hair_texture_desc}
   → Natural flow direction following scalp curvature
   → Appropriate thickness for {hair_type.lower()} characteristics

🎯 Hair Color: {hair_color}
   → Create {color_desc}
   → Match existing hair color if visible in original image
   → Natural color variation and depth

🎯 Hair Density: {hair_density * 100:.1f}% coverage
   → {timeframe_config['maturity']} appropriate for {timeframe} growth
   → {final_multiplier:.1f}x enhancement multiplier for visible results
   → {"AGGRESSIVE DENSITY for dramatic early results" if timeframe == "3months" else "MATURE DENSITY for full restoration results" if timeframe == "8months" else "Gradual density variation for natural appearance"}

🎯 Pattern Integration: {pattern_type}
   → Follow natural hairline patterns
   → Respect facial proportions and symmetry
   → Create age-appropriate hairline shape

PRECISE GENERATION INSTRUCTIONS:
1. EXAMINE Image 2 (the mask) - identify ALL WHITE REGIONS
2. GENERATE thick, dense {hair_type.lower()} ONLY in WHITE masked regions
3. CREATE {color_desc} with natural shine and depth
4. ENSURE {hair_density * 100:.1f}% density coverage within WHITE regions
5. BLEND seamlessly with existing hair at WHITE/BLACK boundaries
6. MAINTAIN natural hair growth direction and flow patterns
7. PRESERVE all BLACK masked areas completely unchanged
8. DO NOT generate hair outside WHITE regions
9. CRITICAL: Use Image 2 as a STRICT TEMPLATE - white = hair, black = no hair

{"ENHANCED 3-MONTH GENERATION REQUIREMENTS:" if timeframe == "3months" else "ENHANCED 8-MONTH GENERATION REQUIREMENTS:" if timeframe == "8months" else "STANDARD GENERATION REQUIREMENTS:"}
{"✅ Generate MAXIMUM VISIBLE hair growth - this is early stage restoration" if timeframe == "3months" else "✅ Generate FULL, MATURE hair restoration - this is advanced stage" if timeframe == "8months" else "✅ Generate natural progressive hair growth"}
{"✅ Create THICK, PROMINENT hair strands for dramatic early results" if timeframe == "3months" else "✅ Create DENSE, FULL-BODIED hair with mature thickness and volume" if timeframe == "8months" else "✅ Create natural hair thickness appropriate for timeframe"}
{"✅ Use ENHANCED density multiplier for clearly visible improvement" if timeframe == "3months" else "✅ Use MAXIMUM density multiplier for complete restoration appearance" if timeframe == "8months" else "✅ Use appropriate density for natural progression"}
{"✅ Focus on DRAMATIC VISIBILITY - patient needs to see clear progress" if timeframe == "3months" else "✅ Focus on COMPLETE RESTORATION - show full hair recovery results" if timeframe == "8months" else "✅ Focus on natural, mature hair growth"}
{"✅ Generate hair that is CLEARLY DISTINGUISHABLE from balding areas" if timeframe == "3months" else "✅ Generate hair that COMPLETELY COVERS and TRANSFORMS balding areas" if timeframe == "8months" else "✅ Generate natural hair progression"}

CRITICAL SUCCESS FACTORS:
✅ Generate VISIBLE, THICK hair growth in ALL WHITE masked areas
✅ Natural {hair_type.lower()} texture and realistic appearance
✅ Consistent {hair_color} coloring throughout new hair
✅ Realistic density matching {hair_density * 100:.1f}% specification
✅ Seamless integration at WHITE/BLACK mask boundaries
✅ Professional hair restoration quality
✅ NO modifications to BLACK masked areas

RESULT: {"Generate a realistic photo showing DRAMATIC 3-month hair restoration progress with CLEARLY VISIBLE new hair growth in " + pattern_type + " pattern. Show THICK, PROMINENT " + hair_type.lower() + " in " + hair_color + " color at " + str(hair_density * 100) + "% density in WHITE masked regions. The result must show OBVIOUS improvement from balding areas." if timeframe == "3months" else "Generate a realistic photo showing COMPLETE 8-month hair restoration with FULL, MATURE hair coverage in " + pattern_type + " pattern. Show DENSE, FULL-BODIED " + hair_type.lower() + " in " + hair_color + " color at " + str(hair_density * 100) + "% density in WHITE masked regions. The result must show COMPLETE TRANSFORMATION from balding to full hair coverage." if timeframe == "8months" else "Generate a realistic photo showing successful " + pattern_type + " hair restoration with " + hair_type.lower() + " in " + hair_color + " color at " + str(hair_density * 100) + "% density in WHITE masked regions only."}

Execute {"AGGRESSIVE 3-month" if timeframe == "3months" else "COMPLETE 8-month" if timeframe == "8months" else "standard"} hair restoration generation now."""

        logger.info(f"HAIRLINE-PROMPT-{request_id}: Final prompt built ({len(complete_prompt)} chars)")
        logger.info(f"HAIRLINE-PROMPT-{request_id}: Enhancement multiplier: {final_multiplier:.1f}x")
        logger.info(f"HAIRLINE-PROMPT-{request_id}: PROMPT CONTENT:")
        logger.info(f"HAIRLINE-PROMPT-{request_id}: {complete_prompt}")
        
        return complete_prompt
        
    except Exception as e:
        logger.error(f"MASK-PROMPT-{request_id}: Error building prompt: {str(e)}")
        # Fallback prompt
        return f"Generate enhanced hairline restoration using the provided mask image as guidance. Apply {hair_density * 100:.1f}% density with {hair_type} characteristics and {hair_color} color for {timeframe} timeframe."

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
            
            # Remove small noise/points by applying morphological operations
            from scipy import ndimage
            
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
        
        # ENHANCED: Apply morphological operations to clean up the mask
        import cv2
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
    """FIXED: Generate hairline using mask-based approach with corrected image extraction"""
    logger.info(f"MASK-GEN-{request_id}: Starting mask-based hairline generation")
    logger.info(f"  - Timeframe: {timeframe}")
    logger.info(f"  - Density: {hair_density * 100:.1f}%")
    logger.info(f"  - Pattern Type: {pattern_type}")
    logger.info(f"  - Hair Type: {hair_type}")
    logger.info(f"  - Hair Color: {hair_color}")
    logger.info(f"  - Mask size: {len(mask_image_data)} bytes")
    logger.info(f"  - Input image size: {len(input_image_data)} bytes")
    
    # Build mask-based prompt using prompts.json configuration
    prompt = build_mask_based_hairline_prompt(timeframe, hair_density, hair_type, hair_color, pattern_type, request_id,
                                            len(input_image_data), len(mask_image_data), 0, 0)
    
    # ENHANCED: Convert pattern image to white mask for better generation
    enhanced_mask_data = convert_pattern_to_white_mask(mask_image_data, request_id)
    
    # Get mask statistics for prompt
    try:
        pattern_image = Image.open(io.BytesIO(mask_image_data))
        gray_pattern = pattern_image.convert('L')
        pattern_array = np.array(gray_pattern)
        white_mask = np.where(pattern_array > 50, 255, 0).astype(np.uint8)
        white_pixel_count = int(np.sum(white_mask == 255))
        total_pixels = int(white_mask.size)
        
        # Update prompt with actual mask statistics
        prompt = build_mask_based_hairline_prompt(timeframe, hair_density, hair_type, hair_color, pattern_type, request_id,
                                                len(input_image_data), len(enhanced_mask_data), white_pixel_count, total_pixels)
    except Exception as e:
        logger.warning(f"Could not get mask statistics: {str(e)}")
        # Use prompt without statistics
        prompt = build_mask_based_hairline_prompt(timeframe, hair_density, hair_type, hair_color, pattern_type, request_id,
                                                len(input_image_data), len(enhanced_mask_data), 0, 0)
    
    # Save mask images for debugging/inspection
    try:
        # Save original mask
        original_mask_path = os.path.join("logs", "saved_settings", f"{request_id}_{timeframe}_original_mask.png")
        with open(original_mask_path, "wb") as f:
            f.write(mask_image_data)
        
        # Save enhanced mask
        enhanced_mask_path = os.path.join("logs", "saved_settings", f"{request_id}_{timeframe}_enhanced_mask.png")
        with open(enhanced_mask_path, "wb") as f:
            f.write(enhanced_mask_data)
            
        logger.info(f"  📁 Saved mask files for inspection:")
        logger.info(f"    - Original: {original_mask_path}")
        logger.info(f"    - Enhanced: {enhanced_mask_path}")
        
        # Update prompt with file paths for debugging
        prompt = prompt.replace("- Request ID: " + request_id, 
                              f"- Request ID: {request_id}\n- Original mask file: {original_mask_path}\n- Enhanced mask file: {enhanced_mask_path}")
        
    except Exception as e:
        logger.warning(f"  ⚠️ Could not save mask files for inspection: {str(e)}")
    
    # Create content for Gemini API with both original image and enhanced mask
    import base64
    
    # Convert bytes to base64 strings for Gemini API
    image_b64 = base64.b64encode(input_image_data).decode('utf-8')
    mask_b64 = base64.b64encode(enhanced_mask_data).decode('utf-8')
    
    # Log the exact prompt being used for generation
    logger.info(f"GENERATION-PROMPT-{request_id}: EXACT PROMPT BEING SENT TO GEMINI:")
    logger.info(f"GENERATION-PROMPT-{request_id}: " + "="*80)
    logger.info(f"GENERATION-PROMPT-{request_id}: {prompt}")
    logger.info(f"GENERATION-PROMPT-{request_id}: " + "="*80)
    
    # FIXED: Use proper content structure
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
    
    # Try generation with multiple models and retries
    max_retries = 3
    models_to_try = ["gemini-2.5-flash-image-preview"]
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                logger.info(f"MASK-GEN-{request_id}: Attempt {attempt + 1}/{max_retries} - Sending to {model_name}...")
                
                model = genai.GenerativeModel(model_name)
                
                # Log the content structure for debugging
                logger.info(f"MASK-GEN-{request_id}: Content structure: {len(content)} items")
                
                response = await asyncio.to_thread(model.generate_content, content)
                
                # FIXED: Proper response parsing for image generation
                logger.info(f"MASK-GEN-{request_id}: Response type: {type(response).__name__}")
                
                # Check primary response structure
                if hasattr(response, 'parts') and response.parts:
                    logger.info(f"MASK-GEN-{request_id}: Parts count: {len(response.parts)}")
                    
                    for i, part in enumerate(response.parts):
                        logger.info(f"MASK-GEN-{request_id}: Part {i}: {type(part).__name__}")
                        
                        # Try to get image data from inline_data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                image_data = part.inline_data.data
                                
                                # Handle both string and bytes formats
                                if isinstance(image_data, str):
                                    try:
                                        decoded_data = base64.b64decode(image_data)
                                        logger.info(f"MASK-GEN-{request_id}: ✓ Generated {len(decoded_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                        return decoded_data
                                    except Exception as decode_error:
                                        logger.error(f"MASK-GEN-{request_id}: Base64 decode error: {decode_error}")
                                        continue
                                elif isinstance(image_data, bytes):
                                    logger.info(f"MASK-GEN-{request_id}: ✓ Generated {len(image_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                    return image_data
                        
                        # Try to get image data from direct data attribute
                        elif hasattr(part, 'data') and part.data:
                            image_data = part.data
                            if isinstance(image_data, bytes):
                                logger.info(f"MASK-GEN-{request_id}: ✓ Generated {len(image_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                return image_data
                            elif isinstance(image_data, str):
                                try:
                                    decoded_data = base64.b64decode(image_data)
                                    logger.info(f"MASK-GEN-{request_id}: ✓ Generated {len(decoded_data)/1024:.1f}KB image on attempt {attempt + 1}")
                                    return decoded_data
                                except Exception as decode_error:
                                    logger.error(f"MASK-GEN-{request_id}: Base64 decode error: {decode_error}")
                                    continue
                
                # Check alternative response structure with candidates
                elif hasattr(response, 'candidates') and response.candidates:
                    logger.info(f"MASK-GEN-{request_id}: Found candidates: {len(response.candidates)}")
                    
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                            image_data = part.inline_data.data
                                            if isinstance(image_data, str):
                                                try:
                                                    decoded_data = base64.b64decode(image_data)
                                                    logger.info(f"MASK-GEN-{request_id}: ✓ Generated {len(decoded_data)/1024:.1f}KB image from candidates")
                                                    return decoded_data
                                                except Exception as decode_error:
                                                    logger.error(f"MASK-GEN-{request_id}: Candidates decode error: {decode_error}")
                                                    continue
                                            elif isinstance(image_data, bytes):
                                                logger.info(f"MASK-GEN-{request_id}: ✓ Generated {len(image_data)/1024:.1f}KB image from candidates")
                                                return image_data
                
                logger.warning(f"MASK-GEN-{request_id}: No valid image data found in response parts")
                    
            except Exception as e:
                logger.error(f"MASK-GEN-{request_id}: Attempt {attempt + 1} failed with {model_name}: {str(e)}")
                logger.error(f"MASK-GEN-{request_id}: Exception type: {type(e).__name__}")
                if "Could not convert" in str(e):
                    logger.error(f"MASK-GEN-{request_id}: This appears to be a successful generation but with extraction issues")
                if attempt == max_retries - 1:
                    logger.error(f"MASK-GEN-{request_id}: All attempts failed with {model_name}")
                else:
                    await asyncio.sleep(1)  # Brief delay before retry
    
    logger.error(f"MASK-GEN-{request_id}: All models and attempts failed")
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
    """Generate different hairline patterns and scalp regions based on detected face"""
    logger.info(f"HAIRLINE-GEN-{request_id}: Generating hairline patterns...")
    
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
        
        # Hairline M Pattern
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

        # Crown - FILLED GREEN CIRCLE (like M pattern)
        img_crown = image.copy()
        crown_y = int(face_top - (face_width * 0.5))
        crown_center = (center_x, crown_y)
        axes_crown = (int(face_width * 0.4), int(face_width * 0.25))
        
        # Fill the entire ellipse area with GREEN color (like M pattern solid coverage)
        cv2.ellipse(img_crown, crown_center, axes_crown, 0, 0, 360, (0, 255, 0), -1)  # FILLED GREEN ellipse
        
        # Add slight transparency by blending with original
        overlay = img_crown.copy()
        cv2.ellipse(overlay, crown_center, axes_crown, 0, 0, 360, (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, img_crown, 0.7, 0, img_crown)  # 30% green, 70% original
        
        # Draw GREEN ellipse outline for definition
        cv2.ellipse(img_crown, crown_center, axes_crown, 0, 0, 360, (0, 255, 0), 3)  # GREEN outline
        crown_filename = f"{timestamp}_{request_id}_crown.jpg"
        crown_path = os.path.join("detections", crown_filename)
        cv2.imwrite(crown_path, img_crown)
        generated_patterns["crown"] = crown_path

        # Full Scalp - FILLED GREEN CIRCLE (like M pattern)
        img_scalp = image.copy()
        scalp_center = (center_x, int(face_top - face_height * 0.6))
        axes_scalp = (int(face_width * 0.6), int(face_height * 0.9))
        
        # Fill the entire ellipse area with GREEN color (like M pattern solid coverage)
        cv2.ellipse(img_scalp, scalp_center, axes_scalp, 0, 0, 360, (0, 255, 0), -1)  # FILLED GREEN ellipse
        
        # Add slight transparency by blending with original
        overlay = img_scalp.copy()
        cv2.ellipse(overlay, scalp_center, axes_scalp, 0, 0, 360, (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, img_scalp, 0.7, 0, img_scalp)  # 30% green, 70% original
        
        # Draw GREEN ellipse outline for definition
        cv2.ellipse(img_scalp, scalp_center, axes_scalp, 0, 0, 360, (0, 255, 0), 3)  # GREEN outline
        scalp_filename = f"{timestamp}_{request_id}_scalp.jpg"
        scalp_path = os.path.join("detections", scalp_filename)
        cv2.imwrite(scalp_path, img_scalp)
        generated_patterns["full_scalp"] = scalp_path

        # Mid-Crown - FILLED GREEN CIRCLE (like M pattern)
        img_mid_crown = image.copy()
        mid_crown_center = (center_x, int(scalp_center[1] - axes_scalp[1] * 0.5))
        axes_mid = (int(face_width * 0.3), int(face_width * 0.15))
        
        # Fill the entire ellipse area with GREEN color (like M pattern solid coverage)
        cv2.ellipse(img_mid_crown, mid_crown_center, axes_mid, 0, 0, 360, (0, 255, 0), -1)  # FILLED GREEN ellipse
        
        # Add slight transparency by blending with original
        overlay = img_mid_crown.copy()
        cv2.ellipse(overlay, mid_crown_center, axes_mid, 0, 0, 360, (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, img_mid_crown, 0.7, 0, img_mid_crown)  # 30% green, 70% original
        
        # Draw GREEN ellipse outline for definition
        cv2.ellipse(img_mid_crown, mid_crown_center, axes_mid, 0, 0, 360, (0, 255, 0), 3)  # GREEN outline
        mid_crown_filename = f"{timestamp}_{request_id}_mid_crown.jpg"
        mid_crown_path = os.path.join("detections", mid_crown_filename)
        cv2.imwrite(mid_crown_path, img_mid_crown)
        generated_patterns["mid_crown"] = mid_crown_path

        logger.info(f"HAIRLINE-GEN-{request_id}: Generated {len(generated_patterns)} hairline patterns")

        return {
            "success": True,
            "patterns": generated_patterns,
            "pattern_count": len(generated_patterns)
        }

    except Exception as e:
        logger.error(f"HAIRLINE-GEN-{request_id}: Error generating patterns: {str(e)}")
        return {"error": f"Pattern generation error: {str(e)}"}

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
        half_bald_analysis = None
        if detection_result["face_detected"]:
            pattern_result = generate_hairlines_and_scalp_regions(
                image_data, detection_result["face_bounds"], request_id
            )
            
            # ENHANCED: Add half-bald pattern analysis
            half_bald_analysis = detect_half_bald_pattern(
                image_data, detection_result["face_bounds"], request_id
            )
            
            if half_bald_analysis and not half_bald_analysis.get("error"):
                logger.info(f"DETECT-{request_id}: Half-bald analysis: {half_bald_analysis['pattern_type']}, Enhancement: {half_bald_analysis['recommended_enhancement']}")
        
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
            "half_bald_analysis": half_bald_analysis,  # ENHANCED: Include half-bald analysis
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

@app.on_event("startup")
async def startup_event():
    """Load configuration on startup"""
    global PROMPTS_CONFIG
    PROMPTS_CONFIG = load_prompts_config()
    logger.info("STARTUP: Hair Growth API with Face Detection started successfully")

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
    """Save user settings and pattern/mask data for later generation"""
    request_id = generate_request_id()
    logger.info(f"SAVE-SETTINGS-{request_id}: Saving user settings and pattern data")
    
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
        
        # Save masks if provided
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
        
        logger.info(f"SAVE-SETTINGS-{request_id}: Settings saved successfully")
        logger.info(f"  - Original image: {original_path}")
        logger.info(f"  - Settings file: {settings_path}")
        logger.info(f"  - Masks saved: {len(mask_paths)}")
        
        return {
            "success": True,
            "request_id": request_id,
            "message": "Settings and pattern data saved successfully",
            "saved_files": {
                "original_image": original_path,
                "settings": settings_path,
                "masks": mask_paths
            }
        }
        
    except Exception as e:
        logger.error(f"SAVE-SETTINGS-{request_id}: Error saving settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")

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
    """Generate hair using previously saved pattern/mask data"""
    request_id = generate_request_id()
    logger.info(f"GENERATE-SAVED-{request_id}: Generating hair using saved pattern data")
    
    try:
        # Read input image
        input_data = await image.read()
        input_image = Image.open(io.BytesIO(input_data))
        image_width, image_height = input_image.size
        
        # Find the most recent saved settings for this image (simplified approach)
        settings_dir = os.path.join("logs", "saved_settings")
        if not os.path.exists(settings_dir):
            raise HTTPException(status_code=404, detail="No saved settings found. Please save settings first.")
        
        # Get the most recent settings file (in real app, you'd match by image hash or user ID)
        settings_files = [f for f in os.listdir(settings_dir) if f.endswith("_settings.json")]
        if not settings_files:
            raise HTTPException(status_code=404, detail="No saved settings found. Please save settings first.")
        
        # Use the most recent settings file (by modification time)
        settings_files_with_time = []
        for f in settings_files:
            file_path = os.path.join(settings_dir, f)
            mod_time = os.path.getmtime(file_path)
            settings_files_with_time.append((mod_time, f))
        
        # Sort by modification time and get the most recent
        latest_settings_file = sorted(settings_files_with_time)[-1][1]
        settings_path = os.path.join(settings_dir, latest_settings_file)
        
        logger.info(f"  - Found {len(settings_files)} settings files")
        logger.info(f"  - Using most recent: {latest_settings_file}")
        
        with open(settings_path, "r") as f:
            saved_settings = json.load(f)
        
        logger.info(f"  - Using saved settings: {latest_settings_file}")
        logger.info(f"  - Saved hair line type: {saved_settings['hair_line_type']}")
        
        # Log available masks
        mask_paths = saved_settings.get("mask_paths", {})
        logger.info(f"  - Available masks: {list(mask_paths.keys())}")
        for mask_type, mask_path in mask_paths.items():
            exists = os.path.exists(mask_path) if mask_path else False
            logger.info(f"    - {mask_type}: {mask_path} ({'EXISTS' if exists else 'MISSING'})")
        
        # Generate for BOTH timeframes using saved pattern
        if saved_settings["hair_line_type"] == "Hairline":
            mask_paths = saved_settings.get("mask_paths", {})
            
            # ENHANCED: Use saved hairline mask for direct mask-based generation
            if "hairline_mask" in mask_paths and os.path.exists(mask_paths["hairline_mask"]):
                logger.info(f"  - Using saved hairline pattern mask for MASK-BASED generation: {mask_paths['hairline_mask']}")
                
                # Read saved hairline mask
                with open(mask_paths["hairline_mask"], "rb") as f:
                    hairline_mask_data = f.read()
                
                logger.info(f"  - Generating BOTH timeframes using MASK-BASED approach...")
                
                # Generate both timeframes using mask-based generation
                # Use saved density values from settings
                saved_density_3m = saved_settings.get("hair_density_3m", 0.7)
                saved_density_8m = saved_settings.get("hair_density_8m", 0.9)
                saved_pattern = saved_settings.get("hairline_pattern", "M Pattern")
                
                logger.info(f"  - Using saved pattern: {saved_pattern}")
                logger.info(f"  - Using saved densities: 3M={saved_density_3m}, 8M={saved_density_8m}")
                
                generated_image_3m = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, hairline_mask_data,
                    "3months", saved_density_3m, hair_type, hair_color, saved_pattern, request_id
                )
                
                generated_image_8m = await generate_hairline_with_mask_enhanced(
                    input_data, image.content_type, hairline_mask_data,
                    "8months", saved_density_8m, hair_type, hair_color, saved_pattern, request_id
                )
                
                # Check if generation was successful
                if not generated_image_3m:
                    logger.error(f"GENERATE-SAVED-{request_id}: 3-month mask-based generation failed")
                    raise HTTPException(status_code=500, detail="Failed to generate 3-month hair image using mask-based approach")
                    
                if not generated_image_8m:
                    logger.error(f"GENERATE-SAVED-{request_id}: 8-month mask-based generation failed")
                    raise HTTPException(status_code=500, detail="Failed to generate 8-month hair image using mask-based approach")
                
            else:
                logger.info(f"  - No saved hairline mask found, falling back to coordinate-based generation")
                # Fallback: Use coordinate-based generation
                coordinates_data = generate_hairline_coordinates_from_face_detection(
                    faceDetectionData=None,
                    image_width=image_width,
                    image_height=image_height,
                    request_id=request_id,
                    pattern_type=saved_settings.get("hairline_pattern", "M_pattern")
                )
                
                if "error" in coordinates_data:
                    raise HTTPException(status_code=400, detail=coordinates_data['error'])
                
                # Create pattern image from coordinates for visual reference
                pattern_image_data = create_synthetic_marking_from_coordinates(
                    coordinates_data, image_width, image_height, request_id
                )
                pattern_image_path = os.path.join("logs", "saved_settings", f"{request_id}_pattern_image.png")
                with open(pattern_image_path, "wb") as f:
                    f.write(pattern_image_data)
                logger.info(f"  - Created pattern image: {pattern_image_path}")
                
                # Generate both timeframes using coordinate-based approach
                generated_image_3m = await generate_freemark_hair(
                    input_data, image.content_type, coordinates_data,
                    "3months", hair_density_3m, hair_type, hair_color, request_id
                )
                
                generated_image_8m = await generate_freemark_hair(
                    input_data, image.content_type, coordinates_data,
                    "8months", hair_density_8m, hair_type, hair_color, request_id
                )
            
            # Return both images
            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_image_3m if timeframe == "3months" else generated_image_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_image_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_image_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": "dual_timeframe_mask_based" if "hairline_mask" in mask_paths else "dual_timeframe_coordinate_based",
                "used_saved_pattern": True,
                "saved_pattern_type": "hairline_mask" if "hairline_mask" in mask_paths else "hairline_coordinates"
            }
            
        else:
            # FreeMark mode - use saved masks for BOTH timeframes
            mask_paths = saved_settings.get("mask_paths", {})
            
            # Check if both masks exist
            if "mask_3months" not in mask_paths or "mask_8months" not in mask_paths:
                missing = []
                if "mask_3months" not in mask_paths:
                    missing.append("3months")
                if "mask_8months" not in mask_paths:
                    missing.append("8months")
                raise HTTPException(status_code=404, detail=f"Missing saved masks for: {', '.join(missing)}")
            
            # Generate for 3 months using MASK-BASED approach
            logger.info(f"  - Generating 3-month FreeMark image using MASK-BASED generation...")
            mask_3m_path = mask_paths["mask_3months"]
            with open(mask_3m_path, "rb") as f:
                mask_3m_data = f.read()
            
            logger.info(f"    📄 3-month mask file: {mask_3m_path}")
            logger.info(f"    📊 3-month mask size: {len(mask_3m_data)} bytes")
            logger.info(f"    🎯 3-month density: {hair_density_3m}")
            
            generated_image_3m = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_3m_data,
                "3months", hair_density_3m, hair_type, hair_color, "FreeMark", request_id
            )
            
            # Generate for 8 months using MASK-BASED approach
            logger.info(f"  - Generating 8-month FreeMark image using MASK-BASED generation...")
            mask_8m_path = mask_paths["mask_8months"]
            with open(mask_8m_path, "rb") as f:
                mask_8m_data = f.read()
            
            logger.info(f"    📄 8-month mask file: {mask_8m_path}")
            logger.info(f"    📊 8-month mask size: {len(mask_8m_data)} bytes")
            logger.info(f"    🎯 8-month density: {hair_density_8m}")
            
            generated_image_8m = await generate_hairline_with_mask_enhanced(
                input_data, image.content_type, mask_8m_data,
                "8months", hair_density_8m, hair_type, hair_color, "FreeMark", request_id
            )
            
            # Check if generation was successful
            if not generated_image_3m:
                logger.error(f"GENERATE-SAVED-{request_id}: 3-month FreeMark generation failed")
                raise HTTPException(status_code=500, detail="Failed to generate 3-month FreeMark image")
                
            if not generated_image_8m:
                logger.error(f"GENERATE-SAVED-{request_id}: 8-month FreeMark generation failed")
                raise HTTPException(status_code=500, detail="Failed to generate 8-month FreeMark image")
            
            return {
                "request_id": request_id,
                "image": base64.b64encode(generated_image_3m if timeframe == "3months" else generated_image_8m).decode("utf-8"),
                "image_3months": base64.b64encode(generated_image_3m).decode("utf-8"),
                "image_8months": base64.b64encode(generated_image_8m).decode("utf-8"),
                "has_both_timeframes": True,
                "generation_mode": "dual_timeframe_freemark_masks",
                "used_saved_pattern": True,
                "saved_pattern_type": "freemark_masks"
            }
        
    except Exception as e:
        logger.error(f"GENERATE-SAVED-{request_id}: Error generating with saved pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend running - Face Detection Enabled"}

# Utility Functions
def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

# Logging Functions
def log_parameters(request_id: str, **params):
    """Log all input parameters for debugging"""
    logger.info(f"PARAMS-{request_id}: Input Parameters:")
    for key, value in params.items():
        logger.info(f"  - {key}: {value}")

def log_generation_context(request_id: str, use_mask: bool, hair_line_type: str):
    """Log generation context"""
    logger.info(f"CONTEXT-{request_id}: Generation Context:")
    logger.info(f"  - Method: {'COORDINATE-based' if use_mask else 'PROMPT-based'}")
    logger.info(f"  - Target: {hair_line_type}")

def log_prompt_details(request_id: str, prompt: str, prompt_type: str, truncate: bool = True):
    """Log the actual prompt being sent with type identification"""
    logger.info(f"PROMPT-{request_id}: Using {prompt_type.upper()} prompt")
    if truncate and len(prompt) > 200:
        truncated_prompt = prompt[:200] + "..."
        logger.info(f"  - Preview: {truncated_prompt}")
        logger.info(f"  - Full length: {len(prompt)} characters")
    else:
        logger.info(f"  - Content: {prompt}")

# Mask Generation Functions (for auto-mask mode)
def generate_hairline_mask(width: int, height: int, request_id: str = "") -> bytes:
    """Generate a mask for hairline restoration"""
    logger.info(f"MASK-{request_id}: Generating hairline mask ({width}x{height})")
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    hairline_height = int(height * 0.25)
    
    points = []
    for x in range(width):
        center_x = width // 2
        distance_from_center = abs(x - center_x) / (width // 2)
        
        if distance_from_center > 0.7:
            curve_offset = int(hairline_height * 0.3 * (distance_from_center - 0.7) / 0.3)
        else:
            curve_offset = int(hairline_height * 0.1 * (1 - distance_from_center))
        
        y = hairline_height + curve_offset
        points.append((x, y))
    
    polygon_points = [(0, 0), (width, 0)] + points + [(width, 0), (0, 0)]
    draw.polygon(polygon_points, fill=255)
    
    temple_width = int(width * 0.15)
    temple_height = int(height * 0.35)
    
    draw.ellipse([0, 0, temple_width, temple_height], fill=255)
    draw.ellipse([width - temple_width, 0, width, temple_height], fill=255)
    
    logger.info(f"MASK-{request_id}: Hairline mask generated successfully")
    return mask_to_bytes(mask)

def generate_crown_mask(width: int, height: int, request_id: str = "") -> bytes:
    """Generate a mask for crown restoration"""
    logger.info(f"MASK-{request_id}: Generating crown mask ({width}x{height})")
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    center_x = width // 2
    center_y = int(height * 0.3)
    
    crown_radius_x = int(width * 0.25)
    crown_radius_y = int(height * 0.2)
    
    draw.ellipse([
        center_x - crown_radius_x, center_y - crown_radius_y,
        center_x + crown_radius_x, center_y + crown_radius_y
    ], fill=255)
    
    vertex_center_y = int(height * 0.45)
    vertex_radius_x = int(width * 0.2)
    vertex_radius_y = int(height * 0.15)
    
    draw.ellipse([
        center_x - vertex_radius_x, vertex_center_y - vertex_radius_y,
        center_x + vertex_radius_x, vertex_center_y + vertex_radius_y
    ], fill=255)
    
    logger.info(f"MASK-{request_id}: Crown mask generated successfully")
    return mask_to_bytes(mask)

def generate_mid_crown_mask(width: int, height: int, request_id: str = "") -> bytes:
    """Generate a mask for mid-crown restoration"""
    logger.info(f"MASK-{request_id}: Generating mid-crown mask ({width}x{height})")
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    center_x = width // 2
    center_y = int(height * 0.35)
    
    radius_x = int(width * 0.3)
    radius_y = int(height * 0.25)
    
    draw.ellipse([
        center_x - radius_x, center_y - radius_y,
        center_x + radius_x, center_y + radius_y
    ], fill=255)
    
    logger.info(f"MASK-{request_id}: Mid-crown mask generated successfully")
    return mask_to_bytes(mask)

def generate_full_scalp_mask(width: int, height: int, request_id: str = "") -> bytes:
    """Generate a mask for full scalp restoration"""
    logger.info(f"MASK-{request_id}: Generating full scalp mask ({width}x{height})")
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    center_x = width // 2
    center_y = int(height * 0.35)
    radius_x = int(width * 0.4)
    radius_y = int(height * 0.35)
    
    draw.ellipse([
        center_x - radius_x, center_y - radius_y,
        center_x + radius_x, center_y + radius_y
    ], fill=255)
    
    hairline_height = int(height * 0.25)
    points = []
    for x in range(width):
        center_x_line = width // 2
        distance_from_center = abs(x - center_x_line) / (width // 2)
        
        if distance_from_center > 0.7:
            curve_offset = int(hairline_height * 0.3 * (distance_from_center - 0.7) / 0.3)
        else:
            curve_offset = int(hairline_height * 0.1 * (1 - distance_from_center))
        
        y = hairline_height + curve_offset
        points.append((x, y))
    
    polygon_points = [(0, 0), (width, 0)] + points + [(width, 0), (0, 0)]
    draw.polygon(polygon_points, fill=255)
    
    temple_width = int(width * 0.18)
    temple_height = int(height * 0.4)
    
    draw.ellipse([0, 0, temple_width, temple_height], fill=255)
    draw.ellipse([width - temple_width, 0, width, temple_height], fill=255)
    
    logger.info(f"MASK-{request_id}: Full scalp mask generated successfully")
    return mask_to_bytes(mask)

def mask_to_bytes(mask: Image.Image) -> bytes:
    """Convert PIL Image mask to bytes"""
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format='PNG')
    return mask_bytes.getvalue()

def generate_fallback_hairline_mask(width: int, height: int, request_id: str = "") -> bytes:
    """Generate fallback hairline mask when no face is detected"""
    logger.info(f"FALLBACK-MASK-{request_id}: Generating fallback hairline mask ({width}x{height})")
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Use top 25% of image for hairline area
    hairline_height = int(height * 0.25)
    
    # Create a curved hairline shape
    points = []
    for x in range(width):
        center_x = width // 2
        distance_from_center = abs(x - center_x) / (width // 2)
        
        # Create M-shaped hairline curve
        if distance_from_center > 0.7:
            curve_offset = int(hairline_height * 0.4 * (distance_from_center - 0.7) / 0.3)
        else:
            curve_offset = int(hairline_height * 0.15 * (1 - distance_from_center))
        
        y = hairline_height + curve_offset
        points.append((x, y))
    
    # Fill the hairline area
    polygon_points = [(0, 0), (width, 0)] + points + [(width, 0), (0, 0)]
    draw.polygon(polygon_points, fill=255)
    
    # Add temple areas
    temple_width = int(width * 0.15)
    temple_height = int(height * 0.3)
    
    draw.ellipse([0, 0, temple_width, temple_height], fill=255)
    draw.ellipse([width - temple_width, 0, width, temple_height], fill=255)
    
    logger.info(f"FALLBACK-MASK-{request_id}: Fallback hairline mask generated")
    return mask_to_bytes(mask)

def generate_fallback_crown_mask(width: int, height: int, request_id: str = "") -> bytes:
    """Generate fallback crown mask when no face is detected"""
    logger.info(f"FALLBACK-MASK-{request_id}: Generating fallback crown mask ({width}x{height})")
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Use center-top area for crown
    center_x = width // 2
    center_y = int(height * 0.3)
    
    crown_radius_x = int(width * 0.25)
    crown_radius_y = int(height * 0.2)
    
    draw.ellipse([
        center_x - crown_radius_x, center_y - crown_radius_y,
        center_x + crown_radius_x, center_y + crown_radius_y
    ], fill=255)
    
    logger.info(f"FALLBACK-MASK-{request_id}: Fallback crown mask generated")
    return mask_to_bytes(mask)

def generate_hairline_coordinates_from_face_detection(faceDetectionData, image_width: int, image_height: int, request_id: str = "", pattern_type: str = "M_pattern") -> dict:
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

def create_synthetic_marking_from_coordinates(coordinates_data: dict, width: int, height: int, request_id: str = "") -> bytes:
    """Create a synthetic marking image from coordinate data matching the detected pattern"""
    logger.info(f"SYNTHETIC-{request_id}: Creating synthetic marking image ({width}x{height})")
    
    try:
        # Create black image
        marking = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(marking)
        
        # Get pattern type and hairline points
        pattern_type = coordinates_data.get('hairline_pattern', 'M_pattern')
        hairline_points = coordinates_data.get('hairline_points', [])
        regions = coordinates_data.get('regions', [])
        
        logger.info(f"  - Creating {pattern_type} synthetic marking with {len(hairline_points)} points")
        
        # Draw the hairline pattern as connected lines (matching frontend display)
        if len(hairline_points) >= 2:
            # Convert points to integers
            points = [(int(x), int(y)) for x, y in hairline_points]
            
            if pattern_type == "Z_pattern":
                # Draw Z pattern: top-left → top-right → bottom-left → bottom-right
                if len(points) >= 4:
                    draw.line([points[0], points[1]], fill=(255, 255, 255), width=8)  # Top line
                    draw.line([points[1], points[2]], fill=(255, 255, 255), width=8)  # Diagonal
                    draw.line([points[2], points[3]], fill=(255, 255, 255), width=8)  # Bottom line
            else:
                # Draw M-pattern or Curved pattern as connected line
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i + 1]], fill=(255, 255, 255), width=8)
        
        # Draw white regions for each coordinate area (for generation targeting)
        for i, region in enumerate(regions):
            coords = region.get('coordinates', [])
            if len(coords) >= 3:
                # Convert coordinates to tuples
                polygon_coords = [(int(x), int(y)) for x, y in coords]
                # Use semi-transparent white for regions
                draw.polygon(polygon_coords, fill=(200, 200, 200))
                
                # Add a bright center point for each region
                center_x = sum(x for x, y in polygon_coords) // len(polygon_coords)
                center_y = sum(y for x, y in polygon_coords) // len(polygon_coords)
                draw.ellipse([center_x - 8, center_y - 8, center_x + 8, center_y + 8], fill=(255, 255, 255))
        
        logger.info(f"SYNTHETIC-{request_id}: Created {pattern_type} synthetic marking with {len(regions)} regions")
        
        # Convert to bytes
        marking_bytes = io.BytesIO()
        marking.save(marking_bytes, format='PNG')
        return marking_bytes.getvalue()
        
    except Exception as e:
        logger.error(f"SYNTHETIC-{request_id}: Error creating synthetic marking: {str(e)}")
        # Return a simple white rectangle as fallback
        fallback = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(fallback)
        draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], fill=(255, 255, 255))
        fallback_bytes = io.BytesIO()
        fallback.save(fallback_bytes, format='PNG')
        return fallback_bytes.getvalue()

def generate_mask_for_hair_line_type(hair_line_type: str, width: int, height: int, request_id: str = "", use_fallback: bool = False) -> bytes:
    """Generate appropriate mask based on hair line type"""
    if use_fallback:
        # Use fallback generators for when no face is detected
        fallback_generators = {
            "Hairline": generate_fallback_hairline_mask,
            "Crown": generate_fallback_crown_mask,
            "Mid Crown": generate_fallback_crown_mask,  # Use crown mask as fallback
            "Full Scalp": generate_fallback_hairline_mask  # Use hairline mask as fallback
        }
        
        if hair_line_type not in fallback_generators:
            logger.error(f"FALLBACK-MASK-{request_id}: Unsupported hair line type: {hair_line_type}")
            raise ValueError(f"Unsupported hair line type: {hair_line_type}")
        
        return fallback_generators[hair_line_type](width, height, request_id)
    else:
        # Use original face-based generators
        mask_generators = {
            "Hairline": generate_hairline_mask,
            "Crown": generate_crown_mask,
            "Mid Crown": generate_mid_crown_mask,
            "Full Scalp": generate_full_scalp_mask
        }
        
        if hair_line_type not in mask_generators:
            logger.error(f"MASK-{request_id}: Unsupported hair line type: {hair_line_type}")
            raise ValueError(f"Unsupported hair line type: {hair_line_type}")
        
        return mask_generators[hair_line_type](width, height, request_id)

def log_and_save_image(image_data: bytes, filename: str, stage: str, 
                       use_coordinates: bool, request_id: str, timeframe: str = None) -> str:
    """FIXED: Save image with proper sequencing for coordinate-based generation"""
    try:
        os.makedirs("image_logs", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.'))
        method_str = "_coords" if use_coordinates else "_mask"
        
        if stage == "input":
            saved_filename = f"{timestamp}_{request_id}_1_INPUT_{safe_filename}{method_str}.png"
        elif stage.startswith("marking"):
            saved_filename = f"{timestamp}_{request_id}_2_MARKING_{safe_filename}{method_str}.png"
        elif stage == "mask":
            saved_filename = f"{timestamp}_{request_id}_2_MASK_{safe_filename}{method_str}.png"
        elif stage == "output":
            timeframe_clean = timeframe.replace(" ", "").replace("months", "m").replace("Months", "m") if timeframe else "result"
            saved_filename = f"{timestamp}_{request_id}_3_OUTPUT_{timeframe_clean}{method_str}.png"
        else:
            saved_filename = f"{timestamp}_{request_id}_{stage.upper()}{method_str}.png"
        
        saved_path = os.path.join("image_logs", saved_filename)
        
        with open(saved_path, "wb") as f:
            f.write(image_data)
            
        file_size = len(image_data) / 1024  # KB
        logger.info(f"FILE-{request_id}: Saved {stage.upper()}: {os.path.basename(saved_path)} ({file_size:.1f} KB)")
        return saved_path
        
    except Exception as e:
        logger.error(f"FILE-{request_id}: Failed to save {stage}: {str(e)}")
        return None

def validate_generation_quality(image_data: bytes, request_id: str) -> dict:
    """Basic quality validation for generated images"""
    try:
        if not image_data or len(image_data) < 1000:
            logger.warning(f"QUALITY-{request_id}: Image too small or empty ({len(image_data)} bytes)")
            return {"valid": False, "reason": "Image too small or empty"}
        
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        
        if width < 100 or height < 100:
            logger.warning(f"QUALITY-{request_id}: Image dimensions too small ({width}x{height})")
            return {"valid": False, "reason": "Image dimensions too small"}
            
        image.verify()
        
        file_size = len(image_data) / 1024  # KB
        logger.info(f"QUALITY-{request_id}: Generation passed validation ({width}x{height}, {file_size:.1f} KB)")
        return {"valid": True, "dimensions": f"{width}x{height}", "size_kb": f"{file_size:.1f}"}
        
    except Exception as e:
        logger.error(f"QUALITY-{request_id}: Validation failed: {str(e)}")
        return {"valid": False, "reason": f"Validation error: {str(e)}"}

def detect_half_bald_pattern(image_data: bytes, face_bounds: dict, request_id: str = "") -> dict:
    """Detect half-bald patterns and suggest optimal generation strategy"""
    logger.info(f"HALF-BALD-{request_id}: Analyzing hair loss pattern...")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Could not decode image for half-bald analysis"}
        
        h, w, _ = image.shape
        
        # Define scalp regions based on face bounds
        face_top = int(face_bounds["top"])
        face_width = int(face_bounds["width"])
        face_center_x = int(face_bounds["center_x"])
        
        # Analyze different scalp regions
        regions = {
            "hairline": (0, max(0, face_top - 60), w, face_top),
            "crown": (face_center_x - face_width//3, max(0, face_top - face_width//2), 
                     face_center_x + face_width//3, face_top),
            "temples": [(0, face_top - 40, face_width//4, face_top + 20),
                       (w - face_width//4, face_top - 40, w, face_top + 20)]
        }
        
        hair_analysis = {}
        
        for region_name, bounds in regions.items():
            if region_name == "temples":
                # Analyze both temples
                temple_scores = []
                for i, temple_bounds in enumerate(bounds):
                    x1, y1, x2, y2 = temple_bounds
                    roi = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if roi.size > 0:
                        score = analyze_hair_density_in_region(roi)
                        temple_scores.append(score)
                hair_analysis[region_name] = sum(temple_scores) / len(temple_scores) if temple_scores else 0
            else:
                x1, y1, x2, y2 = bounds
                roi = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if roi.size > 0:
                    hair_analysis[region_name] = analyze_hair_density_in_region(roi)
                else:
                    hair_analysis[region_name] = 0
        
        # Determine half-bald pattern
        avg_density = sum(hair_analysis.values()) / len(hair_analysis)
        is_half_bald = avg_density < 0.3  # Threshold for sparse hair
        
        pattern_type = "normal"
        if is_half_bald:
            if hair_analysis["hairline"] < 0.2:
                pattern_type = "receding_hairline"
            elif hair_analysis["crown"] < 0.2:
                pattern_type = "crown_thinning"
            elif hair_analysis["temples"] < 0.2:
                pattern_type = "temple_recession"
            else:
                pattern_type = "diffuse_thinning"
        
        logger.info(f"HALF-BALD-{request_id}: Pattern detected: {pattern_type}, Average density: {avg_density:.2f}")
        
        return {
            "is_half_bald": bool(is_half_bald),
            "pattern_type": pattern_type,
            "average_density": float(avg_density),
            "region_analysis": {k: float(v) for k, v in hair_analysis.items()},
            "recommended_enhancement": "aggressive" if is_half_bald else "standard"
        }
        
    except Exception as e:
        logger.error(f"HALF-BALD-{request_id}: Analysis error: {str(e)}")
        return {"error": f"Half-bald analysis failed: {str(e)}"}

def analyze_hair_density_in_region(roi_image):
    """Analyze hair density in a region of interest"""
    if roi_image.size == 0:
        return 0
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Use texture analysis to detect hair
    # Hair typically has more texture/variation than scalp
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize the variance to a 0-1 scale (higher = more hair-like texture)
    # These thresholds are empirically determined
    if laplacian_var > 500:
        return min(1.0, laplacian_var / 1000)
    else:
        return laplacian_var / 500

# Enhanced standard generation prompt for half-bald optimization
def build_standard_generation_prompt(timeframe: str, hair_type: str, hair_density: float, 
                                   hair_color: str, hair_line_type: str, request_id: str = "") -> str:
    """ENHANCED: Build standard generation prompt with half-bald optimization from prompts.json"""
    logger.info(f"PROMPT-{request_id}: Building enhanced standard prompt for {hair_line_type}")
    
    # Load prompts configuration
    global PROMPTS_CONFIG
    if not PROMPTS_CONFIG:
        PROMPTS_CONFIG = load_prompts_config()
    
    standard_config = PROMPTS_CONFIG.get("generation_prompts", {}).get("standard_generation", {})
    
    # Enhanced timeframe-specific description from prompts.json
    timeframe_specs = standard_config.get("timeframe_specs", {})
    if timeframe in ["3months", "3 months", "3 Months"]:
        timeframe_key = "3months"
        enhancement_multiplier = 1.4
        timeframe_desc = f"Enhanced early regrowth with {hair_density:.1%} density + {enhancement_multiplier}x multiplier"
        growth_type = "clearly visible early improvement"
    else:
        timeframe_key = "8months"
        enhancement_multiplier = 1.6
        timeframe_desc = f"Maximum mature growth with {hair_density:.1%} density + {enhancement_multiplier}x multiplier"
        growth_type = "dramatically enhanced established growth"
    
    # Get region-specific configuration
    region_focus = standard_config.get("region_focus", {})
    region_config = region_focus.get(hair_line_type, {})
    region_description = region_config.get("description", f"Enhanced {hair_line_type.lower()} restoration")
    region_multiplier = region_config.get("enhancement_multiplier", 1.3)
    
    # Enhanced hair color from prompts.json
    freemark_config = PROMPTS_CONFIG.get("generation_prompts", {}).get("freemark_generation", {})
    color_instructions = freemark_config.get("color_instructions", {})
    if hair_color in color_instructions:
        color_desc = color_instructions[hair_color]
    elif hair_color == "#000000":
        color_desc = "DRAMATIC jet black with enhanced natural highlights and maximum scalp contrast"
    else:
        color_desc = f"enhanced {hair_color} with dramatic shine and optimal visibility"
    
    # Build enhanced prompt using prompts.json template
    base_template = standard_config.get("base_template", "Generate enhanced hair growth for {region} region")
    
    prompt = f"""ENHANCED HAIR GROWTH SIMULATION - MASK-BASED GENERATION WITH HALF-BALD OPTIMIZATION

SYSTEM ROLE: {standard_config.get('system_role', 'Expert medical illustrator creating enhanced hair restoration')}

TASK: {standard_config.get('task', 'Generate dramatically visible hair growth using mask guidance')}

TARGET AREA: {hair_line_type} region with enhanced visibility requirements
REGION FOCUS: {region_description}

ENHANCED TIMEFRAME SPECIFICATION:
{timeframe_desc}
Maturity level: {growth_type}
Enhancement multiplier: {region_multiplier}x for {hair_line_type}

ENHANCED GENERATION REQUIREMENTS:
- Hair type: {hair_type} with enhanced thickness and natural texture
- Hair color: {color_desc}
- Target density: {hair_density:.1%} + {enhancement_multiplier}x visibility multiplier
- Region enhancement: {region_multiplier}x multiplier for {hair_line_type} area
- Growth pattern: Natural direction with aggressive enhancement for maximum visibility
- Preserve all facial features and skin texture from input image
- Create dramatically visible improvement while maintaining photographic realism
- Apply enhanced density for obvious transformation results

CRITICAL SUCCESS FACTORS FOR HALF-BALD OPTIMIZATION:
1. Generate DRAMATICALLY VISIBLE hair within the provided mask area
2. Apply aggressive enhancement multipliers for maximum visual impact
3. Use optimal contrast and thickness for improved visibility in sparse areas
4. Create obvious improvement that would be clear in before/after comparisons
5. Maintain strict mask boundary compliance while maximizing visible impact
6. Show clear advancement appropriate to {timeframe} timeframe
7. Preserve input image quality in unmasked areas with seamless integration

Generate the maximally visible hair restoration simulation now."""

    logger.info(f"PROMPT-{request_id}: Generated enhanced {len(prompt)} character standard prompt")
    return prompt
    
if __name__ == "__main__":
    import uvicorn
    logger.info("STARTUP: Starting Hair Growth API - Consolidated Individual Endpoint")
    uvicorn.run(app, host="0.0.0.0", port=8000)