import os
import sys
import unittest
import database
from pdf_challan_generator import generate_pdf_challan


class TestP5DashboardComponents(unittest.TestCase):
    """Automated test suite verifying P5 database layer and PDF generation."""

    def setUp(self):
        database.init_db()

    def test_01_database_kpis_real(self):
        """Verifies KPI computation returns real non-null metrics."""
        kpis = database.get_dashboard_kpis()
        self.assertIn("total_vehicles_detected", kpis)
        self.assertIn("total_violations", kpis)
        self.assertIn("helmet_violations", kpis)
        self.assertIn("triple_riding_violations", kpis)
        self.assertIn("verified_number_plates", kpis)
        self.assertIn("total_fine_amount", kpis)
        self.assertGreaterEqual(kpis["total_violations"], 0)
        print(f"PASS: Real KPIs retrieved: {kpis}")

    def test_02_database_analytics_metrics(self):
        """Verifies analytics computation returns structured category breakdowns."""
        analytics = database.get_analytics_metrics()
        self.assertIn("categories", analytics)
        self.assertIn("top_offenders", analytics)
        self.assertIn("sources", analytics)
        self.assertIn("audit_statuses", analytics)
        print(f"PASS: Analytics metrics: categories={analytics['categories']}, sources={analytics['sources']}")

    def test_03_pdf_challan_generation_bytes(self):
        """Verifies generating official PDF e-Challan directly in memory as bytes."""
        sample_record = {
            "challan_id": "TEST-ECH-99999",
            "vehicle_plate": "TS09EA1234",
            "date_time": "21-Nov-2026 01:45:00 PM",
            "rider_count": 3,
            "unhelmeted_count": 2,
            "triple_riding": True,
            "helmet_violation": True,
            "total_fine": 3000,
            "violations": ["Riding Without Helmet (2 Riders)", "Triple Riding on Two-Wheeler"],
            "evidence_path": "final_violation_output.jpg",
            "audit_status": "VERIFIED"
        }
        pdf_bytes = generate_pdf_challan(sample_record)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        print(f"PASS: PDF bytes generated successfully ({len(pdf_bytes)} bytes, starts with %PDF)")

    def test_04_database_insert_search_delete(self):
        """Verifies full lifecycle of a challan record in SQLite database."""
        test_id = "TEST-ECH-LIFECYCLE-001"
        test_plate = "DL01AB9999"

        # 1. Insert
        inserted = database.insert_challan({
            "challan_id": test_id,
            "vehicle_plate": test_plate,
            "date_time": "21-Nov-2026 01:50:00 PM",
            "rider_count": 2,
            "unhelmeted_count": 1,
            "triple_riding": False,
            "helmet_violation": True,
            "total_fine": 1000,
            "violations": ["Riding Without Helmet (1 Rider)"],
            "ocr_confidence": 0.95,
            "plate_valid": True,
            "evidence_path": "final_violation_output.jpg",
            "audit_status": "VERIFIED",
            "source_type": "IMAGE"
        })
        self.assertTrue(inserted, "Failed to insert test challan into SQLite")

        # 2. Search by plate
        found = database.get_all_challans(search_plate="DL01")
        matching = [c for c in found if c["challan_id"] == test_id]
        self.assertEqual(len(matching), 1, "Failed to find inserted test challan by plate search")
        self.assertEqual(matching[0]["vehicle_plate"], test_plate)

        # 3. Get by ID
        rec = database.get_challan_by_id(test_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["total_fine"], 1000)

        # 4. Clean up / Delete
        deleted = database.delete_challan(test_id)
        self.assertTrue(deleted)
        rec_after = database.get_challan_by_id(test_id)
        self.assertIsNone(rec_after)
        print("PASS: Database lifecycle (insert, search, get, delete) verified cleanly!")


if __name__ == "__main__":
    unittest.main()
