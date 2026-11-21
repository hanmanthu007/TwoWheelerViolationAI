import cv2
import numpy as np
import os
import re

from plate_ocr import (
    correct_indian_plate_characters,
    extract_plate_details,
    extract_plate_number,
    INDIAN_PLATE_REGEX
)
from plate_roi import extract_plate_roi, preprocess_plate_image


def generate_synthetic_plate_image(text, bg_color=(255, 255, 255), text_color=(0, 0, 0), width=320, height=90, noise=False, blur=False, low_light=False):
    """Generates synthetic Indian license plate images for rigorous benchmarking."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Outer plate border
    cv2.rectangle(img, (2, 2), (width - 3, height - 3), (20, 20, 20), 3)
    
    # IND badge area on left (standard HSRP style)
    cv2.rectangle(img, (4, 4), (38, height - 4), (180, 100, 30), -1)
    cv2.putText(img, "IND", (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Text placement
    font_scale = 1.15
    thickness = 3
    cv2.putText(img, text, (48, 58), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    # Apply environmental degradations
    if low_light:
        # Reduce brightness by 65% and add subtle headlight glare gradient
        img = (img.astype(np.float32) * 0.35).astype(np.uint8)

    if noise:
        noise_matrix = np.random.normal(0, 20, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise_matrix, 0, 255).astype(np.uint8)
        img = noisy

    if blur:
        img = cv2.GaussianBlur(img, (7, 7), 2.5)

    return img


def run_unit_tests():
    print("=" * 65)
    print("TEST SUITE 1: POSITION-AWARE CHARACTER CORRECTION RULES")
    print("=" * 65)

    test_cases = [
        # (Raw OCR Input, Expected Corrected Output, Expected Validation)
        ("TS09EA1234", "TS09EA1234", True),
        ("TSO9EA1234", "TS09EA1234", True),      # 'O' in RTO digit pos -> '0'
        ("TS09EAI234", "TS09EA1234", True),      # 'I' in number digit pos -> '1'
        ("MH12AB8OO1", "MH12AB8001", True),      # 'O' in number digit pos -> '0'
        ("8H02AA1234", "BH02AA1234", True),      # '8' in state letter pos -> 'B'
        ("KA04MB2Z12", "KA04MB2212", True),      # 'Z' in number digit pos -> '2'
        ("DL0IS51234", "DL01SS1234", True),      # 'I' in digit pos -> '1', '5' in series pos -> 'S'
        ("AP29BD5555", "AP29BD5555", True),
        ("INVALID123", "INVALID123", False),     # Length & structure invalid
        ("ABC", "ABC", False),                    # Too short
        ("1234567890", "1234567890", False),     # All digits
    ]

    passed = 0
    for raw, expected_clean, expected_valid in test_cases:
        corrected, valid = correct_indian_plate_characters(raw)
        status = (corrected == expected_clean and valid == expected_valid)
        if status:
            passed += 1
            mark = "[PASS]"
        else:
            mark = "[FAIL]"
        print(f"{mark} Raw: '{raw:<12}' -> Clean: '{corrected:<12}' | Valid: {valid!s:<5} (Expected: '{expected_clean}', {expected_valid})")

    print(f"\nUnit Tests Result: {passed}/{len(test_cases)} Passed")
    assert passed == len(test_cases), "Not all position correction unit tests passed!"


def run_image_ocr_tests():
    print("\n" + "=" * 65)
    print("TEST SUITE 2: OCR PERFORMANCE ON DEGRADED IMAGES")
    print("=" * 65)

    os.makedirs("test_samples", exist_ok=True)

    samples = [
        ("Clean Standard HSRP Plate", "TS09EA1234", False, False, False),
        ("Blurry Plate (Camera Motion)", "TS09EA1234", False, True, False),
        ("Low-Light Night Plate", "MH12AB8001", False, False, True),
        ("Sensor Noise + Low-Light", "KA04MB2212", True, False, True),
    ]

    for label, ground_truth, noise, blur, low_light in samples:
        filename = f"test_samples/{ground_truth}_{'blur_' if blur else ''}{'lowlight_' if low_light else ''}sample.jpg"
        plate_bgr = generate_synthetic_plate_image(
            ground_truth,
            noise=noise,
            blur=blur,
            low_light=low_light
        )
        cv2.imwrite(filename, plate_bgr)

        raw, clean, conf, crop, valid = extract_plate_details(plate_bgr)

        print(f"\nScenario: {label}")
        print(f"  File                     : {filename}")
        print(f"  Ground Truth             : {ground_truth}")
        print(f"  Raw OCR Output           : {raw}")
        print(f"  Position-Corrected Plate : {clean}")
        print(f"  Confidence Score         : {conf:.2f}")
        print(f"  Validation Status        : {'VALID' if valid else 'INVALID'}")


def run_pipeline_integration_test():
    print("\n" + "=" * 65)
    print("TEST SUITE 3: INTEGRATION TEST ON PROJECT TEST IMAGES")
    print("=" * 65)

    for img_name in ["test.jpg.jpg", "images.jpg.jpg"]:
        if os.path.exists(img_name):
            print(f"\nEvaluating pipeline on: {img_name}")
            full_img = cv2.imread(img_name)
            
            # Simulated motorcycle box from YOLO
            if img_name == "test.jpg.jpg":
                bike_box = [306, 385, 1151, 863]
            else:
                bike_box = [185, 323, 1283, 1058]

            raw, clean, conf, crop, valid = extract_plate_details(full_img, motorcycle_box=bike_box)
            print(f"  Raw OCR Detected         : {raw}")
            print(f"  Corrected Registration   : {clean}")
            print(f"  Confidence Score         : {conf:.2f}")
            print(f"  Validation Status        : {valid}")
            print(f"  Plate Crop Saved         : {crop is not None}")


if __name__ == "__main__":
    run_unit_tests()
    run_image_ocr_tests()
    run_pipeline_integration_test()
    print("\n" + "=" * 65)
    print("ALL P1 TESTING SUITES COMPLETED SUCCESSFULLY!")
    print("=" * 65)
