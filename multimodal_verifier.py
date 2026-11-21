from PIL import Image
import cv2
import json
import os
import re
import time

# Supported default Gemini Vision Flash model
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_client():
    """
    Safely retrieves the official Google GenAI client using GEMINI_API_KEY.
    Returns None if the API key is not configured.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[Gemini Init Warning] Failed to initialize Google GenAI client: {e}")
        return None


def verify_plate_with_gemini(
    image_input,
    model_name=DEFAULT_GEMINI_MODEL,
    max_retries=3,
    timeout_seconds=15
):
    """
    Sends a cropped license plate (or vehicle image) to Gemini Vision to read
    the visible registration number and return structured verification data.
    
    Args:
        image_input (str or np.ndarray): File path or OpenCV BGR image array.
        model_name (str): Gemini model identifier.
        max_retries (int): Number of retry attempts on transient 503/429 errors.
        timeout_seconds (int): Timeout per attempt.
        
    Returns:
        dict:
            - 'detected_number': str (e.g. 'TS09EA1234' or 'UNKNOWN')
            - 'confidence': float (0.0 to 1.0)
            - 'is_readable': bool
            - 'explanation': str
            - 'status': 'SUCCESS' | 'GEMINI_UNAVAILABLE' | 'API_ERROR'
            - 'model_used': str
    """
    fallback_response = {
        "detected_number": "UNKNOWN",
        "confidence": 0.0,
        "is_readable": False,
        "explanation": "Gemini verification unavailable (API key not set or service unreachable).",
        "status": "GEMINI_UNAVAILABLE",
        "model_used": model_name
    }

    # 1. Check API Key
    client = get_gemini_client()
    if client is None:
        return fallback_response

    # 2. Convert input image to PIL RGB
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                fallback_response["explanation"] = f"Image file not found: {image_input}"
                fallback_response["status"] = "FILE_NOT_FOUND"
                return fallback_response
            pil_img = Image.open(image_input).convert("RGB")
        elif hasattr(image_input, "shape"):
            # OpenCV BGR image
            rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_arr)
        else:
            fallback_response["explanation"] = "Invalid image input format."
            fallback_response["status"] = "INVALID_INPUT"
            return fallback_response
    except Exception as e:
        fallback_response["explanation"] = f"Image decoding error: {e}"
        fallback_response["status"] = "IMAGE_DECODE_ERROR"
        return fallback_response

    # 3. Prompt requesting structured JSON output
    prompt = """
You are an expert traffic camera AI and vehicle license plate recognition auditor.
Analyze this cropped two-wheeler license plate image.

Read the visible vehicle registration number.
CRITICAL RULES:
- Never invent missing characters or force a valid format.
- If characters are blurry, partially cut off, or illegible, mark is_readable as false and use UNKNOWN for detected_number.
- Return strictly uppercase alphanumeric characters for detected_number with no spaces, dashes, or punctuation.

Return a valid JSON object matching exactly this structure:
{
  "detected_number": "<license plate string in uppercase without spaces, or UNKNOWN>",
  "confidence": <float from 0.0 to 1.0 representing your confidence in this reading>,
  "is_readable": <true if the full registration number is clearly visible and readable, false otherwise>,
  "explanation": "<brief 1-sentence technical observation of plate condition, font, or ambiguity>"
}

Do not include any conversational markdown or explanation outside the JSON object.
"""

    # 4. Multi-model fallback list in case selected model identifier is unavailable
    model_candidates = [
        model_name,
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-3.8-flash"
    ]
    # Deduplicate while preserving order
    seen_models = set()
    model_sequence = [m for m in model_candidates if not (m in seen_models or seen_models.add(m))]

    last_error = ""

    for target_model in model_sequence:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=[pil_img, prompt]
                )

                response_text = (response.text or "").strip()

                # Extract JSON block
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group(0))
                else:
                    parsed_json = json.loads(response_text)

                det_num = str(parsed_json.get("detected_number", "UNKNOWN")).upper().strip()
                det_num = re.sub(r'[^A-Z0-9]', '', det_num) if det_num != "UNKNOWN" else "UNKNOWN"

                conf = float(parsed_json.get("confidence", 0.0))
                is_read = bool(parsed_json.get("is_readable", False))
                expl = str(parsed_json.get("explanation", "Verification completed."))

                return {
                    "detected_number": det_num,
                    "confidence": conf,
                    "is_readable": is_read,
                    "explanation": expl,
                    "status": "SUCCESS",
                    "model_used": target_model
                }

            except Exception as e:
                last_error = str(e)
                if "503" in last_error or "429" in last_error or "UNAVAILABLE" in last_error:
                    wait_time = attempt * 2
                    time.sleep(wait_time)
                else:
                    # Model not found or other non-transient error, try next candidate model
                    break

    fallback_response["explanation"] = f"Gemini API request failed: {last_error}"
    fallback_response["status"] = "API_ERROR"
    return fallback_response


def compare_and_reconcile_plate(easyocr_result, gemini_result):
    """
    Compares and reconciles EasyOCR and Gemini Multimodal results.
    
    Args:
        easyocr_result (dict):
            - 'plate': str
            - 'conf': float
            - 'is_valid': bool
            - 'raw': str
        gemini_result (dict):
            - 'detected_number': str
            - 'confidence': float
            - 'is_readable': bool
            - 'explanation': str
            - 'status': str
            
    Returns:
        dict:
            - 'final_plate': str
            - 'verification_status': 'VERIFIED' | 'REVIEW_REQUIRED' | 'GEMINI_VERIFIED' | 'CONFIRMED_UNREADABLE' | 'LOCAL_OCR_ONLY'
            - 'is_confirmed': bool (True only if verified and valid)
            - 'summary': str
    """
    easy_plate = easyocr_result.get("plate", "PLATE_NOT_DETECTED")
    easy_conf = easyocr_result.get("conf", 0.0)
    easy_valid = easyocr_result.get("is_valid", False)

    gemini_status = gemini_result.get("status", "GEMINI_UNAVAILABLE")
    gemini_plate = gemini_result.get("detected_number", "UNKNOWN")
    gemini_conf = gemini_result.get("confidence", 0.0)
    gemini_readable = gemini_result.get("is_readable", False)
    gemini_expl = gemini_result.get("explanation", "")

    # Case 1: Gemini was unavailable or not invoked
    if gemini_status != "SUCCESS":
        if easy_valid and easy_plate != "PLATE_NOT_DETECTED":
            return {
                "final_plate": easy_plate,
                "verification_status": "LOCAL_OCR_ONLY",
                "is_confirmed": easy_conf >= 0.50,
                "summary": f"Local OCR plate: {easy_plate} (Gemini unavailable)."
            }
        else:
            return {
                "final_plate": "PLATE_NOT_DETECTED",
                "verification_status": "UNCONFIRMED_LOCAL_ONLY",
                "is_confirmed": False,
                "summary": "Local OCR failed or uncertain (Gemini unavailable)."
            }

    # Case 2: Both EasyOCR and Gemini Agree
    if easy_valid and easy_plate != "PLATE_NOT_DETECTED" and gemini_readable and gemini_plate == easy_plate:
        return {
            "final_plate": easy_plate,
            "verification_status": "VERIFIED",
            "is_confirmed": True,
            "summary": f"Dual-engine verified: EasyOCR ({easy_conf:.2f}) and Gemini ({gemini_conf:.2f}) agree on {easy_plate}."
        }

    # Case 3: Both Agree Plate is Unreadable / Obscured
    if (easy_plate in ["PLATE_NOT_DETECTED", "UNKNOWN", ""]) and (not gemini_readable or gemini_plate == "UNKNOWN"):
        return {
            "final_plate": "PLATE_NOT_DETECTED",
            "verification_status": "CONFIRMED_UNREADABLE",
            "is_confirmed": False,
            "summary": f"Dual-engine agreement: Plate is confirmed unreadable. {gemini_expl}"
        }

    # Case 4: Gemini successfully rescued an unread or uncertain plate
    if (not easy_valid or easy_plate == "PLATE_NOT_DETECTED") and gemini_readable and gemini_plate != "UNKNOWN":
        from plate_ocr import INDIAN_PLATE_REGEX, correct_indian_plate_characters
        clean_gem, is_gem_valid = correct_indian_plate_characters(gemini_plate)

        if is_gem_valid and gemini_conf >= 0.65:
            return {
                "final_plate": clean_gem,
                "verification_status": "GEMINI_VERIFIED",
                "is_confirmed": True,
                "summary": f"Gemini multimodal verified valid plate: {clean_gem} ({gemini_conf:.2f}). {gemini_expl}"
            }

    # Case 5: Disagreement between EasyOCR and Gemini
    return {
        "final_plate": easy_plate if easy_valid else gemini_plate,
        "verification_status": "REVIEW_REQUIRED",
        "is_confirmed": False,
        "summary": f"Conflict detected! EasyOCR read '{easy_plate}' vs Gemini read '{gemini_plate}'. Manual review required."
    }


# =========================================================
# BACKWARD COMPATIBLE LEGACY INTERFACE
# =========================================================

def verify_with_gemini(image_input):
    """
    Preserved legacy function for backward compatibility.
    """
    res = verify_plate_with_gemini(image_input)
    plate = res.get("detected_number", "UNKNOWN")
    reason = res.get("explanation", "Verification completed.")
    return plate, reason


if __name__ == "__main__":
    print("=" * 60)
    print("       GEMINI MULTIMODAL NUMBER PLATE VERIFIER (P3)")
    print("=" * 60)
    test_img = "detected_plate_crop.jpg" if os.path.exists("detected_plate_crop.jpg") else "test_with_plate.jpg"
    print(f"Testing input image: {test_img}")

    res = verify_plate_with_gemini(test_img)
    print("\nAPI Response:")
    print(json.dumps(res, indent=2))
    print("=" * 60)
