import os
import io
import unittest
import cv2
import numpy as np
from PIL import Image

from pdf_challan_generator import (
    generate_pdf_challan,
    generate_and_save_pdf,
    generate_demonstration_qr,
    REPORTS_DIR
)


class TestP6AdvancedPdfGenerator(unittest.TestCase):
    """Test suite verifying P6 Advanced PDF Challan Generation under M.V. Act 2019."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def test_01_helmet_violation_pdf(self):
        """Generates and verifies PDF for isolated helmet violation (Sec 129/194D)."""
        record = {
            "challan_id": "DEMO-ECH-HELMET-001",
            "vehicle_plate": "TS09EA1111",
            "date_time": "21-Nov-2026 02:15:00 PM",
            "location": "Junction 01 - Begumpet Main Road",
            "rider_count": 1,
            "unhelmeted_count": 1,
            "triple_riding": False,
            "helmet_violation": True,
            "total_fine": 1000,
            "ocr_confidence": 0.95,
            "plate_valid": True,
            "evidence_path": "final_violation_output.jpg",
            "plate_crop": "detected_plate_crop.jpg",
            "rider_list": [
                {"helmet_status": "NO_HELMET", "head_conf": 0.93}
            ],
            "audit_status": "VERIFIED"
        }

        save_path, pdf_bytes = generate_and_save_pdf(record, "sample_helmet_violation.pdf")
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        print(f"PASS: [Test 1] Helmet Violation PDF generated -> {save_path} ({len(pdf_bytes):,} bytes)")

    def test_02_triple_riding_pdf(self):
        """Generates and verifies PDF for isolated triple riding (Sec 128/194C)."""
        record = {
            "challan_id": "DEMO-ECH-TRIPLE-002",
            "vehicle_plate": "AP09BK5678",
            "date_time": "21-Nov-2026 02:20:00 PM",
            "location": "Sector #07 - Outer Ring Road Expressway",
            "rider_count": 3,
            "unhelmeted_count": 0,
            "triple_riding": True,
            "helmet_violation": False,
            "total_fine": 1000,
            "ocr_confidence": 0.91,
            "plate_valid": True,
            "evidence_path": "final_violation_output.jpg",
            "plate_crop": "detected_plate_crop.jpg",
            "rider_list": [
                {"helmet_status": "HELMET", "head_conf": 0.90},
                {"helmet_status": "HELMET", "head_conf": 0.88},
                {"helmet_status": "HELMET", "head_conf": 0.87}
            ],
            "audit_status": "VERIFIED"
        }

        save_path, pdf_bytes = generate_and_save_pdf(record, "sample_triple_riding.pdf")
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        print(f"PASS: [Test 2] Triple Riding PDF generated -> {save_path} ({len(pdf_bytes):,} bytes)")

    def test_03_multiple_violations_pdf(self):
        """Generates and verifies PDF for multiple violations (Helmet + Triple Riding)."""
        record = {
            "challan_id": "DEMO-ECH-MULTIPLE-003",
            "vehicle_plate": "DL01AB9999",
            "date_time": "21-Nov-2026 02:25:00 PM",
            "location": "Junction 04 - Panjagutta Circle",
            "rider_count": 3,
            "unhelmeted_count": 2,
            "triple_riding": True,
            "helmet_violation": True,
            "total_fine": 3000,
            "ocr_confidence": 0.942,
            "plate_valid": True,
            "evidence_path": "final_violation_output.jpg",
            "plate_crop": "detected_plate_crop.jpg",
            "rider_list": [
                {"helmet_status": "HELMET", "head_conf": 0.89},
                {"helmet_status": "NO_HELMET", "head_conf": 0.94},
                {"helmet_status": "NO_HELMET", "head_conf": 0.91}
            ],
            "audit_status": "VERIFIED"
        }

        save_path, pdf_bytes = generate_and_save_pdf(record, "sample_multiple_violations.pdf")
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        print(f"PASS: [Test 3] Multiple Violations PDF generated -> {save_path} ({len(pdf_bytes):,} bytes)")

    def test_04_missing_images_and_incomplete_record(self):
        """Verifies graceful handling of missing images and incomplete metadata without crash."""
        incomplete_rec = {
            "challan_id": "DEMO-ECH-INCOMPLETE-004",
            "vehicle_plate": "UNKNOWN",
            "evidence_path": "non_existent_image_path_12345.jpg",
            "plate_crop": None,
            "rider_list": []
            # Intentionally omitting other fields to test fallbacks
        }

        save_path, pdf_bytes = generate_and_save_pdf(incomplete_rec, "sample_incomplete_record.pdf")
        self.assertTrue(os.path.exists(save_path))
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        print(f"PASS: [Test 4] Handled missing images & incomplete record cleanly -> {save_path}")

    def test_05_qr_code_scanning_and_content(self):
        """Verifies QR code can be scanned by a standard CV reader and confirms non-payable demo link."""
        qr_buf, demo_url = generate_demonstration_qr("DEMO-ECH-SCAN-005", "TS09EA1234", 2000)
        self.assertIn("status=DEMO_ONLY_NON_PAYABLE", demo_url)

        # Decode using OpenCV QRCodeDetector
        pil_img = Image.open(qr_buf).convert("RGB")
        cv_img = np.array(pil_img)
        detector = cv2.QRCodeDetector()
        decoded_text, points, _ = detector.detectAndDecode(cv_img)

        self.assertTrue(bool(decoded_text), "QRCodeDetector failed to decode QR image")
        self.assertIn("DEMO_ONLY_NON_PAYABLE", decoded_text)
        self.assertIn("TS09EA1234", decoded_text)
        print(f"PASS: [Test 5] QR Code scanned & verified: '{decoded_text[:65]}...'")


if __name__ == "__main__":
    unittest.main()
