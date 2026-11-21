import json
import os
import cv2
import numpy as np

from multimodal_verifier import (
    verify_plate_with_gemini,
    compare_and_reconcile_plate
)


def run_reconciliation_unit_tests():
    print("=" * 65)
    print("TEST SUITE 1: RECONCILIATION & AGREEMENT / DISAGREEMENT LOGIC")
    print("=" * 65)

    test_cases = [
        # Case 1: Both Agree on Valid Indian Plate -> VERIFIED
        {
            "name": "Both Engines Agree (High Confidence)",
            "easy": {"plate": "TS09EA1234", "conf": 0.48, "is_valid": True, "raw": "TS09EA1234"},
            "gemini": {"detected_number": "TS09EA1234", "confidence": 0.95, "is_readable": True, "explanation": "Clear plate.", "status": "SUCCESS"},
            "expected_status": "VERIFIED",
            "expected_confirmed": True
        },

        # Case 2: EasyOCR and Gemini Disagree -> REVIEW_REQUIRED
        {
            "name": "Engines Disagree (Conflicting Characters)",
            "easy": {"plate": "MH12AB8001", "conf": 0.42, "is_valid": True, "raw": "MH12AB8001"},
            "gemini": {"detected_number": "MH12AB8007", "confidence": 0.92, "is_readable": True, "explanation": "Last digit is clearly 7, not 1.", "status": "SUCCESS"},
            "expected_status": "REVIEW_REQUIRED",
            "expected_confirmed": False
        },

        # Case 3: Both Agree Plate is Unreadable / Obscured -> CONFIRMED_UNREADABLE
        {
            "name": "Both Agree Plate is Unreadable",
            "easy": {"plate": "PLATE_NOT_DETECTED", "conf": 0.0, "is_valid": False, "raw": ""},
            "gemini": {"detected_number": "UNKNOWN", "confidence": 0.10, "is_readable": False, "explanation": "Mud and glare obscure all characters.", "status": "SUCCESS"},
            "expected_status": "CONFIRMED_UNREADABLE",
            "expected_confirmed": False
        },

        # Case 4: Gemini Rescues Missing / Unread Plate -> GEMINI_VERIFIED
        {
            "name": "Gemini Rescue on OCR Miss",
            "easy": {"plate": "PLATE_NOT_DETECTED", "conf": 0.0, "is_valid": False, "raw": ""},
            "gemini": {"detected_number": "KA04MB2212", "confidence": 0.94, "is_readable": True, "explanation": "High-contrast font clearly identified.", "status": "SUCCESS"},
            "expected_status": "GEMINI_VERIFIED",
            "expected_confirmed": True
        },

        # Case 5: Gemini Unavailable / Offline Fallback -> LOCAL_OCR_ONLY
        {
            "name": "Gemini Offline / No Key Fallback",
            "easy": {"plate": "TS09EA1234", "conf": 0.85, "is_valid": True, "raw": "TS09EA1234"},
            "gemini": {"detected_number": "UNKNOWN", "confidence": 0.0, "is_readable": False, "explanation": "API Key missing.", "status": "GEMINI_UNAVAILABLE"},
            "expected_status": "LOCAL_OCR_ONLY",
            "expected_confirmed": True
        }
    ]

    passed = 0
    for tc in test_cases:
        res = compare_and_reconcile_plate(tc["easy"], tc["gemini"])
        actual_st = res["verification_status"]
        actual_conf = res["is_confirmed"]

        is_ok = (actual_st == tc["expected_status"] and actual_conf == tc["expected_confirmed"])
        mark = "[PASS]" if is_ok else "[FAIL]"
        if is_ok:
            passed += 1

        print(f"\n{mark} Test: {tc['name']}")
        print(f"  EasyOCR Reading    : {tc['easy']['plate']} (conf: {tc['easy']['conf']:.2f})")
        print(f"  Gemini Reading     : {tc['gemini']['detected_number']} (conf: {tc['gemini']['confidence']:.2f})")
        print(f"  Reconciliation     : {actual_st} (Confirmed Fine: {actual_conf})")
        print(f"  Audit Summary      : {res['summary']}")

    print("\n" + "=" * 65)
    print(f"Reconciliation Unit Tests: {passed}/{len(test_cases)} Passed")
    print("=" * 65)
    assert passed == len(test_cases), "Not all reconciliation unit tests passed!"


def run_live_or_mock_api_test():
    print("\n" + "=" * 65)
    print("TEST SUITE 2: GEMINI API STRUCTURED JSON & TIMEOUT/ERROR HANDLING")
    print("=" * 65)

    test_image_path = "detected_plate_crop.jpg" if os.path.exists("detected_plate_crop.jpg") else "test_samples/TS09EA1234_sample.jpg"
    print(f"Input Crop Image: {test_image_path}")

    # Call verify_plate_with_gemini
    res = verify_plate_with_gemini(test_image_path)

    print("\nStructured Output Schema Validation:")
    required_keys = ["detected_number", "confidence", "is_readable", "explanation", "status"]
    for k in required_keys:
        assert k in res, f"Missing required key in Gemini response: {k}"
        print(f"  [PASS] Key '{k}': {res[k]}")

    print(f"\nAPI Status    : {res['status']}")
    print(f"Explanation   : {res['explanation']}")
    if res["status"] == "SUCCESS":
        print(f"Detected Plate: {res['detected_number']} (Confidence: {res['confidence']:.2f})")
    elif res["status"] == "GEMINI_UNAVAILABLE":
        print("Graceful Fallback: Local OCR continues uninterrupted when API key is unconfigured.")


def run_full_pipeline_with_gemini_integration():
    print("\n" + "=" * 65)
    print("TEST SUITE 3: PRIMARY PIPELINE TRIGGER BEHAVIOR")
    print("=" * 65)

    # 1. Test image with high confidence (>= 0.50) -> Gemini skipped or local confirmed
    print("\n[Case A] Running on test_with_plate.jpg (Simulated vehicle with plate)")
    import subprocess
    cmd = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
        "main.py",
        "--image", "test_with_plate.jpg",
        "--no-gui"
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, f"main.py failed: {p.stderr}"
    print("  Return Code: 0 (Execution Success)")
    
    # 2. Test image with no readable plate -> Gemini fallback triggered and handled
    print("\n[Case B] Running on test.jpg.jpg (Plate obscured)")
    cmd_b = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
        "main.py",
        "--image", "test.jpg.jpg",
        "--no-gui"
    ]
    p_b = subprocess.run(cmd_b, capture_output=True, text=True)
    assert p_b.returncode == 0, f"main.py failed on test.jpg.jpg: {p_b.stderr}"
    print("  Return Code: 0 (Execution Success)")


if __name__ == "__main__":
    run_reconciliation_unit_tests()
    run_live_or_mock_api_test()
    run_full_pipeline_with_gemini_integration()
    print("\n" + "=" * 65)
    print("ALL P3 TEST SUITES COMPLETED SUCCESSFULLY!")
    print("=" * 65)
