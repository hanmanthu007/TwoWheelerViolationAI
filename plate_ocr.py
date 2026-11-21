import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import easyocr
import cv2
import re
import os
import numpy as np

# Import plate ROI and preprocessing module
from plate_roi import extract_plate_roi, preprocess_plate_image

print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR initialized successfully!\n")

# Allowed standard license plate characters
PLATE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Standard Indian Vehicle Registration Regex
INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')

# Official Indian State & Union Territory Codes (including BH for Bharat Series)
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UA", "UP", "WB", "BH"
}

# Common OCR confusion mappings
CHAR_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6'
}

CHAR_TO_ALPHA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G'
}


def clean_text(text):
    """Filters out any non-alphanumeric character and strips HSRP country code affixes (IND)."""
    if not text:
        return ""
    cleaned = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    # Strip common HSRP emblem tags ('IND') if attached at start or end
    if len(cleaned) > 10:
        cleaned = re.sub(r'^IND|IND$', '', cleaned)
    return cleaned


def correct_indian_plate_characters(raw_text):
    """
    Applies position-aware character correction for common OCR mistakes:
      - O and 0
      - B and 8
      - I and 1
      - S and 5
      - Z and 2
    
    Correction is applied strictly according to the expected character position
    of the Indian registration format: ^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$
    
    Returns:
        tuple: (corrected_text, is_valid)
    """
    cleaned = clean_text(raw_text)
    
    # Check if already strictly valid and starts with an official Indian state/UT code
    if INDIAN_PLATE_REGEX.match(cleaned) and cleaned[:2] in INDIAN_STATE_CODES:
        return cleaned, True

    # Indian plate lengths range from 8 to 11 characters
    if len(cleaned) < 8 or len(cleaned) > 11:
        return cleaned, False

    # Guard: A valid plate cannot be pure numbers
    if cleaned.isdigit() or sum(c.isdigit() for c in cleaned) >= len(cleaned) - 1:
        return cleaned, False

    # 1. State Code: First 2 characters MUST be letters
    state_part = "".join(CHAR_TO_ALPHA.get(c, c) for c in cleaned[:2])
    if state_part not in INDIAN_STATE_CODES:
        # If the state code does not exist in India, it's not a valid Indian registration
        return cleaned, False

    # 2. Registration Number: Last 4 characters MUST be digits
    number_part = "".join(CHAR_TO_DIGIT.get(c, c) for c in cleaned[-4:])

    # 3. Middle Section: RTO district digits (1 or 2) + Series letters (1, 2, or 3)
    middle = cleaned[2:-4]
    mid_len = len(middle)

    # Try all valid partition points where RTO digits are 1-2 and Series letters are 1-3
    best_candidate = None

    for rto_len in [2, 1]:
        series_len = mid_len - rto_len
        if 1 <= series_len <= 3:
            raw_rto = middle[:rto_len]
            raw_series = middle[rto_len:]

            cand_rto = "".join(CHAR_TO_DIGIT.get(c, c) for c in raw_rto)
            cand_series = "".join(CHAR_TO_ALPHA.get(c, c) for c in raw_series)

            candidate = f"{state_part}{cand_rto}{cand_series}{number_part}"

            if INDIAN_PLATE_REGEX.match(candidate):
                best_candidate = candidate
                break

    if best_candidate is not None:
        return best_candidate, True

    return cleaned, False


def _read_and_sequence_text(image_np, min_conf=0.10):
    """
    Executes EasyOCR with alphanumeric constraints and spatially
    sequences multi-line or fragmented characters top-to-bottom, left-to-right.
    """
    if image_np is None or image_np.size == 0:
        return "", 0.0

    try:
        results = reader.readtext(
            image_np,
            allowlist=PLATE_CHARS,
            paragraph=False,
            text_threshold=0.15,
            low_text=0.20,
            mag_ratio=1.2
        )
    except Exception as e:
        print(f"[OCR Warning] EasyOCR read failure: {e}")
        return "", 0.0

    if not results:
        return "", 0.0

    valid_detections = []
    for bbox, raw_detected, conf in results:
        cleaned_tok = clean_text(raw_detected)
        if len(cleaned_tok) > 0 and conf >= min_conf:
            top_left_x = bbox[0][0]
            top_left_y = bbox[0][1]
            valid_detections.append({
                "text": cleaned_tok,
                "conf": conf,
                "x": top_left_x,
                "y": top_left_y
            })

    if not valid_detections:
        return "", 0.0

    # Group characters by horizontal row bands (within 25 pixels vertically)
    valid_detections.sort(key=lambda d: (round(d["y"] / 25.0) * 25.0, d["x"]))

    raw_assembled = "".join(d["text"] for d in valid_detections)
    avg_conf = sum(d["conf"] for d in valid_detections) / float(len(valid_detections))

    return raw_assembled, avg_conf


def extract_plate_details(image_input, motorcycle_box=None, min_conf=0.10):
    """
    Comprehensive License Plate Recognition pipeline:
      1. Automatically extracts and pads the plate ROI from the motorcycle box or scene.
      2. Generates enhanced image variants (CLAHE, bilateral filter, sharpening, Otsu).
      3. Performs multi-pass EasyOCR inference.
      4. Selects highest-confidence OCR candidate.
      5. Applies position-aware Indian plate regex correction.
      6. Validates against ^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$.

    Returns:
        tuple: (original_text, corrected_plate, confidence, crop_img, is_valid)
            - original_text (str): Raw OCR detected string before correction.
            - corrected_plate (str): Corrected registration string or 'PLATE_NOT_DETECTED'.
            - confidence (float): Average confidence score (0.0 to 1.0).
            - crop_img (np.ndarray): Cropped and padded license plate image.
            - is_valid (bool): True if valid Indian registration format; False otherwise.
    """
    # 1. Decode input image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            print(f"[OCR Error] File not found: {image_input}")
            return "PLATE_NOT_DETECTED", "PLATE_NOT_DETECTED", 0.0, None, False
        img = cv2.imread(image_input)
    else:
        img = image_input

    if img is None or img.size == 0:
        print("[OCR Error] Could not decode image.")
        return "PLATE_NOT_DETECTED", "PLATE_NOT_DETECTED", 0.0, None, False

    # 2. Extract Plate ROI
    crop_img = None
    if motorcycle_box is not None:
        crop_img, _ = extract_plate_roi(img, motorcycle_box=motorcycle_box, padding_ratio=0.12)

    # If no motorcycle box was provided or ROI extraction failed:
    if crop_img is None:
        # If the input image is already a crop (small dimensions or aspect ratio > 1.5):
        h, w = img.shape[:2]
        if h < 250 or (float(w) / float(h) >= 1.6 and h < 500):
            crop_img = img
        else:
            # Attempt whole-scene plate ROI detection
            crop_img, _ = extract_plate_roi(img, motorcycle_box=None, padding_ratio=0.12)
            if crop_img is None:
                crop_img = img

    # 3. Generate enhanced variants for blurry / low-light images
    variants = preprocess_plate_image(crop_img, target_height=110)

    pass_images = []
    if "enhanced" in variants:
        pass_images.append(("CLAHE_Enhanced", variants["enhanced"]))
    if "sharpened" in variants:
        pass_images.append(("Sharpened", variants["sharpened"]))
    if "resized_bgr" in variants:
        pass_images.append(("Resized_BGR", variants["resized_bgr"]))
    if "otsu" in variants:
        pass_images.append(("Otsu_Threshold", variants["otsu"]))

    # Fallback to raw crop if preprocessing dictionary is empty
    if not pass_images:
        pass_images.append(("Raw_Crop", crop_img))

    # 4. Multi-pass OCR Evaluation & Candidate Scoring
    candidates = []

    for pass_name, proc_img in pass_images:
        raw_seq, conf = _read_and_sequence_text(proc_img, min_conf=min_conf)
        if len(raw_seq) >= 4:
            corrected_str, valid_status = correct_indian_plate_characters(raw_seq)

            # Scoring algorithm: Valid Indian format gets heavy priority bonus
            score = conf
            if valid_status:
                score += 10.0  # Priority bonus for strictly valid Indian plate
            if len(corrected_str) == 10:
                score += 0.5  # Standard 10-character Indian plate bonus

            candidates.append({
                "pass": pass_name,
                "raw": raw_seq,
                "corrected": corrected_str,
                "conf": conf,
                "is_valid": valid_status,
                "score": score
            })

    # 5. Select Best Candidate Based on Confidence and Validation
    if candidates:
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        return (
            best["raw"],
            best["corrected"],
            best["conf"],
            crop_img,
            best["is_valid"]
        )

    # If no candidate reached length >= 4
    return "PLATE_NOT_DETECTED", "PLATE_NOT_DETECTED", 0.0, crop_img, False


def extract_plate_number(image_input, min_conf=0.10, motorcycle_box=None, return_details=False):
    """
    Main extraction interface. Fully backwards compatible with existing calls
    expecting (plate_no, conf), while supporting the new 5-value return when
    return_details=True.
    """
    raw_text, corrected_plate, conf, crop_img, is_valid = extract_plate_details(
        image_input,
        motorcycle_box=motorcycle_box,
        min_conf=min_conf
    )

    if return_details:
        return raw_text, corrected_plate, conf, crop_img, is_valid

    # Backwards-compatible 2-tuple return: (plate_string, conf)
    output_plate = corrected_plate if corrected_plate != "PLATE_NOT_DETECTED" else "UNKNOWN"
    return output_plate, conf


# =========================================================
# STANDALONE EXECUTION & VALIDATION TEST
# =========================================================
if __name__ == "__main__":
    test_image = "images.jpg.jpg" if os.path.exists("images.jpg.jpg") else "test.jpg.jpg"

    print("=" * 60)
    print("      INDIAN VEHICLE NUMBER PLATE RECOGNITION (P1)")
    print("=" * 60)
    print(f"Testing input image : {test_image}")

    raw, clean, conf, crop, valid = extract_plate_details(test_image)

    print(f"Original Detected Text     : {raw}")
    print(f"Corrected Registration No  : {clean}")
    print(f"OCR Confidence             : {conf:.2f}")
    print(f"Validation Status          : {'VALID INDIAN REGISTRATION' if valid else 'INVALID / UNCONFIRMED'}")
    print(f"Crop Generated             : {'YES' if crop is not None else 'NO'}")
    print("=" * 60)
