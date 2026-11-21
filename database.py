
import sqlite3
import os
import json
import re
from datetime import datetime

DB_PATH = "traffic_violations.db"


def get_connection():
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database tables and indexes."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS challan_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        challan_id TEXT UNIQUE NOT NULL,
        vehicle_plate TEXT NOT NULL,
        date_time TEXT NOT NULL,
        rider_count INTEGER DEFAULT 1,
        unhelmeted_count INTEGER DEFAULT 0,
        triple_riding BOOLEAN DEFAULT 0,
        helmet_violation BOOLEAN DEFAULT 0,
        total_fine INTEGER DEFAULT 0,
        violations_json TEXT DEFAULT '[]',
        ocr_confidence REAL DEFAULT 0.0,
        plate_valid BOOLEAN DEFAULT 0,
        evidence_path TEXT DEFAULT '',
        audit_status TEXT DEFAULT 'CONFIRMED',
        source_type TEXT DEFAULT 'IMAGE',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_plate
        ON challan_records(vehicle_plate)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_date
        ON challan_records(date_time)
    """)

    conn.commit()
    conn.close()

    # Synchronize legacy text challans
    sync_existing_text_challans()


def insert_challan(record):
    """
    Inserts a new challan record into the database.
    Ignores duplicate records based on challan_id.
    """
    conn = get_connection()
    cursor = conn.cursor()

    violations_str = (
        json.dumps(record.get("violations", []))
        if isinstance(record.get("violations"), (list, dict))
        else str(record.get("violations", "[]"))
    )

    try:
        cursor.execute("""
        INSERT OR IGNORE INTO challan_records (
            challan_id,
            vehicle_plate,
            date_time,
            rider_count,
            unhelmeted_count,
            triple_riding,
            helmet_violation,
            total_fine,
            violations_json,
            ocr_confidence,
            plate_valid,
            evidence_path,
            audit_status,
            source_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("challan_id"),
            record.get("vehicle_plate", "PLATE_NOT_DETECTED"),
            record.get(
                "date_time",
                datetime.now().strftime("%d-%b-%Y %I:%M:%S %p")
            ),
            record.get("rider_count", 1),
            record.get("unhelmeted_count", 0),
            1 if record.get("triple_riding") else 0,
            1 if record.get("helmet_violation") else 0,
            record.get("total_fine", 0),
            violations_str,
            float(record.get("ocr_confidence", 0.0)),
            1 if record.get("plate_valid") else 0,
            record.get("evidence_path", ""),
            record.get("audit_status", "CONFIRMED"),
            record.get("source_type", "IMAGE")
        ))

        conn.commit()
        success = cursor.rowcount > 0

    except Exception as e:
        print(f"[DB Error] Failed to insert challan: {e}")
        success = False

    finally:
        conn.close()

    return success


def get_all_challans(
    search_plate=None,
    violation_filter=None,
    date_filter=None
):
    """Retrieves filtered challan records."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM challan_records WHERE 1=1"
    params = []

    if search_plate:
        query += " AND vehicle_plate LIKE ?"
        params.append(f"%{search_plate.strip().upper()}%")

    if violation_filter and violation_filter != "All":

        if violation_filter == "Triple Riding":
            query += " AND triple_riding = 1"

        elif violation_filter == "Helmet Violation":
            query += " AND helmet_violation = 1"

        elif violation_filter == "Both Violations":
            query += """
                AND triple_riding = 1
                AND helmet_violation = 1
            """

    if date_filter:
        query += " AND date_time LIKE ?"
        params.append(f"%{date_filter}%")

    query += " ORDER BY id DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    conn.close()

    return [dict(r) for r in rows]


def get_challan_by_id(challan_id):
    """Retrieves a single challan by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM challan_records WHERE challan_id = ?",
        (challan_id,)
    )

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def delete_challan(challan_id):
    """Deletes a challan by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM challan_records WHERE challan_id = ?",
        (challan_id,)
    )

    conn.commit()
    deleted = cursor.rowcount > 0

    conn.close()

    return deleted


def get_dashboard_kpis():
    """Computes dashboard KPIs from the database."""
    conn = get_connection()
    cursor = conn.cursor()

    # Total challan records
    cursor.execute("""
        SELECT COUNT(*)
        FROM challan_records
    """)
    total_violations = cursor.fetchone()[0]

    # Distinct identified registration plates
    cursor.execute("""
        SELECT COUNT(DISTINCT UPPER(TRIM(vehicle_plate)))
        FROM challan_records
        WHERE vehicle_plate IS NOT NULL
          AND TRIM(vehicle_plate) != ''
          AND UPPER(TRIM(vehicle_plate)) NOT IN (
              'PLATE_NOT_DETECTED',
              'UNKNOWN'
          )
    """)
    distinct_vehicles = cursor.fetchone()[0]

    # Helmet violations
    cursor.execute("""
        SELECT COUNT(*)
        FROM challan_records
        WHERE helmet_violation = 1
    """)
    helmet_violations = cursor.fetchone()[0]

    # Triple-riding violations
    cursor.execute("""
        SELECT COUNT(*)
        FROM challan_records
        WHERE triple_riding = 1
    """)
    triple_violations = cursor.fetchone()[0]

    # Verified number plates
    cursor.execute("""
        SELECT COUNT(*)
        FROM challan_records
        WHERE plate_valid = 1
    """)
    verified_plates = cursor.fetchone()[0]

    # Total assessed fines
    cursor.execute("""
        SELECT COALESCE(SUM(total_fine), 0)
        FROM challan_records
    """)
    sum_fines = cursor.fetchone()[0]

    conn.close()

    return {
        "total_vehicles_detected": max(
            distinct_vehicles,
            distinct_vehicles + (total_violations - verified_plates)
        ),
        "total_violations": total_violations,
        "helmet_violations": helmet_violations,
        "triple_riding_violations": triple_violations,
        "verified_number_plates": verified_plates,
        "total_fine_amount": sum_fines
    }


def get_analytics_metrics():
    """Generates analytics and distribution metrics."""

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Category breakdown

    cursor.execute("""
        SELECT COUNT(*)
        FROM challan_records
        WHERE helmet_violation = 1
          AND triple_riding = 0
    """)
    only_helmet = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM challan_records
        WHERE triple_riding = 1
          AND helmet_violation = 0
    """)
    only_triple = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM challan_records
        WHERE triple_riding = 1
          AND helmet_violation = 1
    """)
    both_viol = cursor.fetchone()[0]

    # 2. Repeat offenders
    # Only verified plates with at least two violations

    cursor.execute("""
        SELECT
            UPPER(TRIM(vehicle_plate)) AS vehicle_plate,
            COUNT(*) AS count,
            COALESCE(SUM(total_fine), 0) AS total_fines
        FROM challan_records
        WHERE plate_valid = 1
          AND vehicle_plate IS NOT NULL
          AND TRIM(vehicle_plate) != ''
          AND UPPER(TRIM(vehicle_plate)) NOT IN (
              'PLATE_NOT_DETECTED',
              'UNKNOWN'
          )
        GROUP BY UPPER(TRIM(vehicle_plate))
        HAVING COUNT(*) >= 2
        ORDER BY count DESC, total_fines DESC
        LIMIT 5
    """)

    top_offenders = [
        dict(r) for r in cursor.fetchall()
    ]

    # 3. Surveillance source breakdown

    cursor.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(source_type), ''), 'UNKNOWN') AS source_type,
            COUNT(*) AS count
        FROM challan_records
        GROUP BY COALESCE(NULLIF(TRIM(source_type), ''), 'UNKNOWN')
    """)

    source_counts = {
        r["source_type"]: r["count"]
        for r in cursor.fetchall()
    }

    # 4. OCR verification audit status

    cursor.execute("""
        SELECT
            COALESCE(NULLIF(TRIM(audit_status), ''), 'UNKNOWN') AS audit_status,
            COUNT(*) AS count
        FROM challan_records
        GROUP BY COALESCE(NULLIF(TRIM(audit_status), ''), 'UNKNOWN')
    """)

    status_counts = {
        r["audit_status"]: r["count"]
        for r in cursor.fetchall()
    }

    conn.close()

    return {
        "categories": {
            "Helmet Violation Only": only_helmet,
            "Triple Riding Only": only_triple,
            "Combined (Triple + No Helmet)": both_viol
        },
        "top_offenders": top_offenders,
        "sources": source_counts,
        "audit_statuses": status_counts
    }


def sync_existing_text_challans():
    """
    Imports legacy text challans into SQLite.

    Historical text receipts are not automatically considered
    OCR-verified because their OCR verification cannot be established
    from the registration number alone.
    """

    challans_dir = "challans"

    if not os.path.exists(challans_dir):
        return

    conn = get_connection()
    cursor = conn.cursor()

    for fname in os.listdir(challans_dir):

        if not fname.lower().endswith(".txt"):
            continue

        filepath = os.path.join(challans_dir, fname)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            c_id_match = re.search(
                r"Challan ID\s*:\s*([A-Za-z0-9\-]+)",
                content
            )

            date_match = re.search(
                r"Date & Time\s*:\s*(.+)",
                content
            )

            veh_match = re.search(
                r"Vehicle Number\s*:\s*([A-Za-z0-9_]+)",
                content
            )

            riders_match = re.search(
                r"Riders Detected\s*:\s*(\d+)",
                content
            )

            fine_match = re.search(
                r"TOTAL FINE PAYABLE\s*:\s*Rs\s*(\d+)",
                content
            )

            evidence_match = re.search(
                r"EVIDENCE SAVED\s*:\s*(.+)",
                content
            )

            if not c_id_match or not veh_match:
                continue

            c_id = c_id_match.group(1).strip()
            veh = veh_match.group(1).strip().upper()

            dt = (
                date_match.group(1).strip()
                if date_match
                else datetime.now().strftime("%d-%b-%Y %I:%M:%S %p")
            )

            riders = (
                int(riders_match.group(1))
                if riders_match else 1
            )

            fine = (
                int(fine_match.group(1))
                if fine_match else 0
            )

            evidence = (
                evidence_match.group(1).strip()
                if evidence_match
                else "final_violation_output.jpg"
            )

            has_triple = (
                "Triple Riding" in content
                or "Dangerous Overloading" in content
            )

            has_helmet = (
                "Without Helmet" in content
                or "No Helmet" in content
            )

            unhelmeted = riders if has_helmet else 0

            # Historical text records require verification review.
            # Do not assign artificial OCR confidence or VERIFIED status.

            cursor.execute("""
                INSERT OR IGNORE INTO challan_records (
                    challan_id,
                    vehicle_plate,
                    date_time,
                    rider_count,
                    unhelmeted_count,
                    triple_riding,
                    helmet_violation,
                    total_fine,
                    violations_json,
                    ocr_confidence,
                    plate_valid,
                    evidence_path,
                    audit_status,
                    source_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c_id,
                veh,
                dt,
                riders,
                unhelmeted,
                1 if has_triple else 0,
                1 if has_helmet else 0,
                fine,
                json.dumps(["Synchronized Historical Record"]),
                0.0,
                0,
                evidence,
                "PENDING_REVIEW",
                "HISTORICAL"
            ))

        except Exception as e:
            print(f"[Sync Warning] Skipping {fname}: {e}")

    conn.commit()
    conn.close()


# Initialize database automatically when imported
init_db()
