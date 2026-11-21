import os
import sys
import unittest
import cv2
import numpy as np

# Project AI & Enforcement modules
from ultralytics import YOLO
from head_association import associate_riders_with_helmets, annotate_individual_riders
from plate_roi import extract_plate_roi, detect_license_plate_roi
from plate_ocr import extract_plate_details
from multimodal_verifier import verify_plate_with_gemini, compare_and_reconcile_plate
from fine_generator import calculate_and_generate_challan
from pdf_challan_generator import generate_pdf_challan, generate_and_save_pdf
import database


class TestFullEndToEndIntegrationPipeline(unittest.TestCase):
    """
    Complete end-to-end integration test verifying the unified image-to-report workflow:
    YOLOv8 -> Rider Association -> Head Spatial Mapping (P2) -> Plate ROI & OCR (P1) ->
    Multimodal Reconciliation (P3) -> SQLite Persistence (P5) -> PDF & QR Generation (P6).
    """

    @classmethod
    def setUpClass(cls):
        database.init_db()

    def test_complete_image_to_report_workflow(self):
        print("\n" + "=" * 65)
        print("  STARTING COMPLETE END-TO-END IMAGE-TO-REPORT INTEGRATION TEST")
        print("=" * 65)

        test_image_path = "test_with_plate.jpg"
        if not os.path.exists(test_image_path):
            test_image_path = "test.jpg.jpg"

        self.assertTrue(os.path.exists(test_image_path), f"Input test image not found: {test_image_path}")
        image_bgr = cv2.imread(test_image_path)
        self.assertIsNotNone(image_bgr, "Failed to load test image with OpenCV")
        print(f"[Step 1/8] Input test image loaded: {test_image_path} (shape: {image_bgr.shape})")

        # Step 2: YOLOv8 Detection
        v_model = YOLO("yolov8n.pt")
        h_model = YOLO("models/helmet_model.pt")

        v_res = v_model(image_bgr, conf=0.25, verbose=False)
        self.assertGreater(len(v_res), 0)

        raw_persons = []
        motorcycles = []
        for b in v_res[0].boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            label = v_model.names[cls]
            coords = list(map(int, b.xyxy[0]))
            if label == "person":
                raw_persons.append({"box": coords, "conf": conf})
            elif label == "motorcycle":
                motorcycles.append({"box": coords, "conf": conf, "riders": []})

        print(f"[Step 2/8] YOLOv8 Vehicle Detection: {len(raw_persons)} persons, {len(motorcycles)} motorcycles")

        # Step 3: Spatial Rider-to-Bike Association
        for p in raw_persons:
            px1, py1, px2, py2 = p["box"]
            pcx = (px1 + px2) / 2.0
            p_bottom = py2
            matched_bike = None
            max_overlap = 0

            for bike in motorcycles:
                bx1, by1, bx2, by2 = bike["box"]
                bw = bx2 - bx1
                is_x = (bx1 - 0.10 * bw) <= pcx <= (bx2 + 0.10 * bw)
                is_y = (p_bottom >= by1) and (py1 <= by2)
                if is_x and is_y:
                    overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
                    if overlap_x > max_overlap:
                        max_overlap = overlap_x
                        matched_bike = bike

            if matched_bike is not None:
                matched_bike["riders"].append(p)

        primary_bike = motorcycles[0] if motorcycles else None
        riders_to_evaluate = primary_bike["riders"] if primary_bike else raw_persons
        print(f"[Step 3/8] Spatial Association: {len(riders_to_evaluate)} riders associated to primary motorcycle")

        # Step 4: P2 Spatial Head Association (Upper 30%)
        h_res = h_model(image_bgr, conf=0.30, verbose=False)
        helmet_detections = []
        for b in h_res[0].boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            label = h_model.names[cls].lower().strip()
            coords = list(map(int, b.xyxy[0]))
            helmet_detections.append({
                "box": coords,
                "conf": conf,
                "label": label,
                "is_violation": ("without" in label or "no" in label)
            })

        evaluated_riders = associate_riders_with_helmets(riders_to_evaluate, helmet_detections, head_ratio=0.30)
        rider_count = len(evaluated_riders)
        unhelmeted_count = sum(1 for r in evaluated_riders if r.get("helmet_status") == "NO_HELMET")
        print(f"[Step 4/8] P2 Spatial Head Mapping: {rider_count} riders, {unhelmeted_count} without helmet")

        # Step 5: P1 License Plate ROI Extraction & Position-Aware OCR
        bike_box = primary_bike["box"] if primary_bike else None
        raw_text, corrected_plate, ocr_conf, plate_crop, is_valid_plate = extract_plate_details(
            image_bgr,
            motorcycle_box=bike_box,
            min_conf=0.10
        )
        print(f"[Step 5/8] P1 Plate Extraction: Raw='{raw_text}', Corrected='{corrected_plate}', Valid={is_valid_plate}, Conf={ocr_conf:.2f}")

        # Step 6: P3 Multimodal Verification Fallback
        final_plate = corrected_plate
        audit_tag = "VERIFIED" if is_valid_plate else "REVIEW_REQUIRED"
        if ocr_conf < 0.50 or not is_valid_plate or corrected_plate == "PLATE_NOT_DETECTED":
            gem_in = plate_crop if (plate_crop is not None and getattr(plate_crop, "size", 0) > 0) else image_bgr
            gem_res = verify_plate_with_gemini(gem_in)
            reconciled = compare_and_reconcile_plate(
                easyocr_result={"plate": corrected_plate, "conf": ocr_conf, "is_valid": is_valid_plate, "raw": raw_text},
                gemini_result=gem_res
            )
            final_plate = reconciled["final_plate"]
            audit_tag = reconciled["verification_status"]
            print(f"[Step 6/8] P3 Gemini Fallback executed: Reconciled Plate='{final_plate}', Status={audit_tag}")
        else:
            print(f"[Step 6/8] P3 Gemini Fallback bypassed (EasyOCR confidence {ocr_conf:.2f} >= 0.50 and valid plate)")

        # Step 7: Penalty Calculation & SQLite Database Persistence
        triple_riding = (rider_count >= 3)
        helmet_viol = (unhelmeted_count > 0)
        total_fine = 0
        violations_list = []

        if triple_riding:
            fine = 2000 if rider_count > 3 else 1000
            violations_list.append((f"Triple Riding / Overloading ({rider_count} Persons) - Sec 128/194C MV Act", fine))
            total_fine += fine
        if helmet_viol:
            fine = unhelmeted_count * 1000
            violations_list.append((f"Riding Without Helmet ({unhelmeted_count} Riders) - Sec 129/194D MV Act", fine))
            total_fine += fine

        challan_id = f"DEMO-ECH-INTEGRATION-{os.urandom(2).hex().upper()}"
        evidence_output_path = "final_violation_output.jpg"

        record = {
            "challan_id": challan_id,
            "vehicle_plate": final_plate,
            "date_time": "21-Nov-2026 02:45:00 PM",
            "rider_count": rider_count,
            "unhelmeted_count": unhelmeted_count,
            "triple_riding": triple_riding,
            "helmet_violation": helmet_viol,
            "total_fine": total_fine,
            "violations": violations_list,
            "ocr_confidence": ocr_conf,
            "plate_valid": is_valid_plate,
            "evidence_path": evidence_output_path,
            "plate_crop": plate_crop,
            "rider_list": evaluated_riders,
            "audit_status": audit_tag,
            "source_type": "INTEGRATION_TEST"
        }

        inserted = database.insert_challan(record)
        self.assertTrue(inserted, "Failed to insert integration record into SQLite database")
        print(f"[Step 7/8] SQLite Storage: Inserted record '{challan_id}' (Total Fine: Rs {total_fine})")

        # Step 8: P6 Advanced PDF & Non-Payable QR Code Generation
        pdf_path, pdf_bytes = generate_and_save_pdf(record, f"{challan_id}_{final_plate}.pdf")
        self.assertTrue(os.path.exists(pdf_path), f"PDF file not saved to disk: {pdf_path}")
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        print(f"[Step 8/8] P6 PDF & QR Generation: Output created at {pdf_path} ({len(pdf_bytes):,} bytes)")

        # Verify Database Retrieval & Clean Up
        retrieved = database.get_challan_by_id(challan_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["challan_id"], challan_id)
        self.assertEqual(retrieved["vehicle_plate"], final_plate)

        # Cleanup test record from DB
        database.delete_challan(challan_id)
        print("PASS: Complete Image-to-Report Workflow Integration Test passed 100%!")


if __name__ == "__main__":
    unittest.main()
