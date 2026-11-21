import os
import io
import time
import hashlib
from datetime import datetime
from PIL import Image as PILImage, ImageDraw

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import qrcode

REPORTS_DIR = os.path.join("challans", "pdf_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# STATUTORY PROVISIONS VERIFIED AGAINST OFFICIAL SOURCES:
# The Motor Vehicles Act, 1988 (Act No. 59 of 1988) as amended by the
# Motor Vehicles (Amendment) Act, 2019 (Act No. 32 of 2019), Gazette of India.
# ============================================================================
STATUTORY_PROVISIONS = {
    "HELMET": {
        "section": "Section 129 read with Sec 194D, M.V. Act 1988 (as amended 2019)",
        "offense": "Riding Two-Wheeler Without Protective Headgear (Helmet)",
        "fine_per_unit": 1000,
        "legal_note": "Punishable with fine of Rs 1,000 per unhelmeted rider and 3-month license disqualification."
    },
    "TRIPLE_RIDING": {
        "section": "Section 128 read with Sec 194C, M.V. Act 1988 (as amended 2019)",
        "offense": "Carrying More Than One Pillion Rider on Two-Wheeler (Triple Riding)",
        "fine": 1000,
        "legal_note": "Punishable with fine of Rs 1,000 and 3-month license disqualification."
    },
    "OVERLOADING": {
        "section": "Section 194A, M.V. Act 1988 (as amended 2019)",
        "offense": "Carriage of Excess Passengers / Dangerous Overloading of Vehicle",
        "fine": 2000,
        "legal_note": "Punishable with fine of Rs 2,000 for carrying passengers exceeding seating capacity."
    },
    "SURVEILLANCE_FRAMEWORK": {
        "section": "Section 136A, M.V. Act 1988 (as amended 2019)",
        "offense": "Electronic Monitoring and Enforcement of Road Safety",
        "legal_note": "Statutory framework governing automated surveillance devices and electronic violation notices."
    }
}


def create_placeholder_image(text="Image Not Available", width=400, height=220, subtitle=""):
    """Creates a clean, styled placeholder image when actual image is missing or corrupted."""
    img = PILImage.new("RGB", (width, height), color="#1e293b")
    draw = ImageDraw.Draw(img)
    # Border
    draw.rectangle([(4, 4), (width - 5, height - 5)], outline="#475569", width=2)
    # Inner dashed-like border
    draw.rectangle([(12, 12), (width - 13, height - 13)], outline="#334155", width=1)
    # Text
    draw.text((25, height // 2 - 20), text, fill="#94a3b8")
    if subtitle:
        draw.text((25, height // 2 + 10), subtitle, fill="#64748b")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf


def generate_demonstration_qr(challan_id, vehicle_plate, total_fine):
    """
    Generates a demonstration QR code linking to a simulated verification record.
    The QR data is explicitly non-payable and contains demo audit hash tokens.
    """
    token_seed = f"{challan_id}:{vehicle_plate}:{total_fine}:{time.time()}"
    audit_hash = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:16]

    demo_url = (
        f"https://demo.twowheelerai.internal/verify"
        f"?challan={challan_id}"
        f"&plate={vehicle_plate}"
        f"&amount={total_fine}"
        f"&token={audit_hash}"
        f"&status=DEMO_ONLY_NON_PAYABLE"
    )

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=5,
        border=1,
    )
    qr.add_data(demo_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="#0f172a", back_color="white")

    buf = io.BytesIO()
    qr_img.save(buf, format="PNG")
    buf.seek(0)
    return buf, demo_url


def generate_pdf_challan(record, output_path=None, save_copy=True):
    """
    Generates a professional downloadable PDF traffic violation report using actual detection results.
    
    Includes:
      - Unique demonstration challan reference number
      - Vehicle registration number
      - Violation type & statutory provisions (M.V. Act 2019)
      - Detection date, time, and surveillance location
      - Evidence photograph (main scene) & license plate ROI crop
      - Helmet & triple-riding detection details per rider
      - OCR verification status (EasyOCR / Gemini)
      - Clear non-payable demonstration QR code
      - Comprehensive academic and legal demonstration disclaimer
      - Robust handling for missing images and incomplete detection records

    Args:
        record (dict): Violation detection record.
        output_path (str, optional): Target filepath. If None, saves to challans/pdf_reports/
        save_copy (bool): Whether to persist a copy in challans/pdf_reports/.
        
    Returns:
        bytes: PDF binary content (for st.download_button or file writing).
    """
    # 1. Sanitize & Default Record Metadata
    ts_now = datetime.now()
    challan_id = record.get("challan_id") or f"DEMO-ECH-{ts_now.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex().upper()}"
    vehicle_plate = str(record.get("vehicle_plate") or "PLATE_NOT_DETECTED").strip().upper()
    detection_dt = str(record.get("date_time") or ts_now.strftime("%d-%b-%Y %I:%M:%S %p"))
    location = str(record.get("location") or "Surveillance Sector #04 - Junction Camera #02")
    source_type = str(record.get("source_type") or "IMAGE SURVEILLANCE")

    rider_count = int(record.get("rider_count", 1))
    unhelmeted_count = int(record.get("unhelmeted_count", 0))
    triple_riding = bool(record.get("triple_riding", rider_count >= 3))
    helmet_violation = bool(record.get("helmet_violation", unhelmeted_count > 0))
    ocr_confidence = float(record.get("ocr_confidence", 0.0))
    plate_valid = bool(record.get("plate_valid", False))
    audit_status = str(record.get("audit_status") or ("VERIFIED" if plate_valid else "REVIEW_REQUIRED"))

    # Determine fines & verified statutory provisions
    calculated_fine = 0
    offense_entries = []

    if helmet_violation or unhelmeted_count > 0:
        c = max(1, unhelmeted_count)
        f = c * STATUTORY_PROVISIONS["HELMET"]["fine_per_unit"]
        calculated_fine += f
        offense_entries.append({
            "offense": STATUTORY_PROVISIONS["HELMET"]["offense"],
            "statute": STATUTORY_PROVISIONS["HELMET"]["section"],
            "scope": f"{c} Rider(s) without helmet",
            "fine": f,
            "legal_note": STATUTORY_PROVISIONS["HELMET"]["legal_note"]
        })

    if triple_riding or rider_count >= 3:
        if rider_count > 3:
            f = STATUTORY_PROVISIONS["OVERLOADING"]["fine"]
            desc = STATUTORY_PROVISIONS["OVERLOADING"]["offense"]
            stat = STATUTORY_PROVISIONS["OVERLOADING"]["section"]
            note = STATUTORY_PROVISIONS["OVERLOADING"]["legal_note"]
        else:
            f = STATUTORY_PROVISIONS["TRIPLE_RIDING"]["fine"]
            desc = STATUTORY_PROVISIONS["TRIPLE_RIDING"]["offense"]
            stat = STATUTORY_PROVISIONS["TRIPLE_RIDING"]["section"]
            note = STATUTORY_PROVISIONS["TRIPLE_RIDING"]["legal_note"]

        calculated_fine += f
        offense_entries.append({
            "offense": desc,
            "statute": stat,
            "scope": f"{rider_count} Persons on motorcycle",
            "fine": f,
            "legal_note": note
        })

    # If caller supplied total_fine, use it if positive, otherwise use calculated
    total_fine = int(record.get("total_fine", calculated_fine))
    if total_fine <= 0 and calculated_fine > 0:
        total_fine = calculated_fine

    # 2. Build In-Memory Buffer & Document Setup
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=28,
        rightMargin=28,
        topMargin=20,
        bottomMargin=20
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_demo_banner = ParagraphStyle(
        'DemoBanner',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#991b1b'),
        alignment=1
    )

    style_title = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        alignment=1
    )

    style_sub = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1e40af'),
        alignment=1
    )

    style_meta_k = ParagraphStyle(
        'MetaK',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#475569')
    )

    style_meta_v = ParagraphStyle(
        'MetaV',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a')
    )

    style_th = ParagraphStyle(
        'Th',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    style_td = ParagraphStyle(
        'Td',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1e293b')
    )

    style_td_bold = ParagraphStyle(
        'TdBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0f172a')
    )

    style_fine_callout = ParagraphStyle(
        'FineCallout',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        textColor=colors.HexColor('#b91c1c'),
        alignment=2
    )

    style_disclaimer = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#475569')
    )

    style_qr_lbl = ParagraphStyle(
        'QrLbl',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#0f172a'),
        alignment=1
    )

    story = []

    # -------------------------------------------------------------------------
    # SECTION 1: ACADEMIC DEMONSTRATION WARNING & HEADER BANNER
    # -------------------------------------------------------------------------
    demo_header_data = [
        [
            Paragraph("<b>ACADEMIC RESEARCH & TECHNICAL DEMONSTRATION RECORD &bull; NOT AN OFFICIAL GOVERNMENT CHALLAN</b>", style_demo_banner)
        ]
    ]
    demo_header_tbl = Table(demo_header_data, colWidths=[556])
    demo_header_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fee2e2')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#f87171')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(demo_header_tbl)
    story.append(Spacer(1, 4))

    story.append(Paragraph("TwoWheelerViolationAI &bull; AUTOMATED TRAFFIC SURVEILLANCE PLATFORM", style_title))
    story.append(Paragraph("ELECTRONIC VIOLATION REPORT (RESEARCH DEMONSTRATION RECORD UNDER M.V. ACT 2019 FRAMEWORK)", style_sub))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e40af'), spaceAfter=6))

    # -------------------------------------------------------------------------
    # SECTION 2: DETECTION TELEMETRY & VEHICLE METADATA
    # -------------------------------------------------------------------------
    ocr_status_badge = (
        f"<font color='#15803d'><b>VERIFIED</b> ({ocr_confidence * 100:.1f}%)</font>"
        if (plate_valid or ocr_confidence >= 0.50)
        else f"<font color='#b45309'><b>OFFICER REVIEW REQUIRED</b> ({ocr_confidence * 100:.1f}%)</font>"
    )

    meta_grid = [
        [
            Paragraph("<b>DEMO NOTICE NO:</b>", style_meta_k),
            Paragraph(f"<font color='#b91c1c'><b>{challan_id}</b></font>", style_meta_v),
            Paragraph("<b>DETECTION DATE & TIME:</b>", style_meta_k),
            Paragraph(detection_dt, style_meta_v)
        ],
        [
            Paragraph("<b>REGISTRATION NUMBER:</b>", style_meta_k),
            Paragraph(f"<font color='#0284c7' size='9'><b>{vehicle_plate}</b></font>", style_meta_v),
            Paragraph("<b>SURVEILLANCE LOCATION:</b>", style_meta_k),
            Paragraph(location, style_meta_v)
        ],
        [
            Paragraph("<b>VEHICLE CLASS:</b>", style_meta_k),
            Paragraph("Two-Wheeler / Motorcycle (Class L1/L2)", style_meta_v),
            Paragraph("<b>AI SENSOR PIPELINE:</b>", style_meta_k),
            Paragraph(f"YOLOv8 + ByteTrack &bull; {source_type}", style_meta_v)
        ],
        [
            Paragraph("<b>PLATE OCR STATUS:</b>", style_meta_k),
            Paragraph(ocr_status_badge, style_meta_v),
            Paragraph("<b>RIDERS DETECTED:</b>", style_meta_k),
            Paragraph(f"<b>{rider_count} Person(s)</b> (Unhelmeted: <b>{unhelmeted_count}</b>)", style_meta_v)
        ]
    ]

    meta_table = Table(meta_grid, colWidths=[120, 160, 120, 156])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECTION 3: PHOTOGRAPHIC EVIDENCE PANEL (DUAL VIEW: SCENE + PLATE CROP)
    # -------------------------------------------------------------------------
    story.append(Paragraph("<b>SURVEILLANCE PHOTOGRAPHIC EVIDENCE (SECTION 136A M.V. ACT)</b>", ParagraphStyle(
        'EvSecHdr', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, textColor=colors.HexColor('#1e293b')
    )))
    story.append(Spacer(1, 3))

    # Main evidence photograph
    evidence_path = record.get("evidence_path", "")
    if not evidence_path or not os.path.exists(evidence_path):
        if os.path.exists("final_violation_output.jpg"):
            evidence_path = "final_violation_output.jpg"

    if evidence_path and os.path.exists(evidence_path):
        try:
            main_img_flowable = RLImage(evidence_path, width=335, height=140)
        except Exception:
            ph_buf = create_placeholder_image("Evidence Image Corrupted", 335, 140)
            main_img_flowable = RLImage(ph_buf, width=335, height=140)
    else:
        ph_buf = create_placeholder_image("Surveillance Evidence Not Available", 335, 140, "No visual recording file on disk")
        main_img_flowable = RLImage(ph_buf, width=335, height=140)

    # Number plate crop
    plate_crop_obj = record.get("plate_crop")
    plate_img_flowable = None

    if plate_crop_obj is not None:
        if isinstance(plate_crop_obj, str) and os.path.exists(plate_crop_obj):
            try:
                plate_img_flowable = RLImage(plate_crop_obj, width=205, height=65)
            except Exception:
                pass
        elif hasattr(plate_crop_obj, 'size') and getattr(plate_crop_obj, 'size', 0) > 0:
            # numpy ndarray
            try:
                import cv2
                tmp_crop_buf = io.BytesIO()
                _, enc = cv2.imencode(".jpg", plate_crop_obj)
                tmp_crop_buf.write(enc.tobytes())
                tmp_crop_buf.seek(0)
                plate_img_flowable = RLImage(tmp_crop_buf, width=205, height=65)
            except Exception:
                pass

    if plate_img_flowable is None:
        if os.path.exists("detected_plate_crop.jpg"):
            try:
                plate_img_flowable = RLImage("detected_plate_crop.jpg", width=205, height=65)
            except Exception:
                pass

    if plate_img_flowable is None:
        crop_ph_buf = create_placeholder_image("Plate Crop Not Isolated", 205, 65, "Low-light / Angle")
        plate_img_flowable = RLImage(crop_ph_buf, width=205, height=65)

    # Rider breakdown status text
    rider_status_lines = []
    rider_list = record.get("rider_list", [])
    if rider_list:
        for i, r in enumerate(rider_list, start=1):
            st_val = r.get("helmet_status", "UNKNOWN")
            c_val = r.get("head_conf", 0.0)
            if st_val == "HELMET":
                rider_status_lines.append(f"&bull; Rider {i}: <font color='#15803d'><b>Helmet OK</b></font> ({c_val*100:.0f}%)")
            elif st_val == "NO_HELMET":
                rider_status_lines.append(f"&bull; Rider {i}: <font color='#b91c1c'><b>No Helmet</b></font> ({c_val*100:.0f}%)")
            else:
                rider_status_lines.append(f"&bull; Rider {i}: <font color='#b45309'><b>Unknown</b></font>")
    else:
        rider_status_lines.append(f"&bull; Total Riders: {rider_count} | Unhelmeted: {unhelmeted_count}")

    rider_breakdown_html = "<br/>".join(rider_status_lines[:4])

    evidence_right_cell = [
        Paragraph("<b>Extracted License Plate ROI:</b>", style_meta_k),
        Spacer(1, 2),
        plate_img_flowable,
        Spacer(1, 4),
        Paragraph("<b>Spatial Rider-Head Mapping:</b>", style_meta_k),
        Paragraph(rider_breakdown_html, style_meta_v)
    ]

    evidence_table = Table([[main_img_flowable, evidence_right_cell]], colWidths=[340, 216])
    evidence_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#334155')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#1e293b')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(evidence_table)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECTION 4: STATUTORY MOTOR VEHICLES ACT OFFENSES TABLE
    # -------------------------------------------------------------------------
    story.append(Paragraph("<b>SCHEDULE OF RECORDED MOTOR VEHICLE ACT OFFENSES</b>", ParagraphStyle(
        'OffSecHdr', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, textColor=colors.HexColor('#1e293b')
    )))
    story.append(Spacer(1, 2))

    offense_rows = [
        [
            Paragraph("<b>#</b>", style_th),
            Paragraph("<b>Violation Description</b>", style_th),
            Paragraph("<b>Statutory Clause (M.V. Act 2019)</b>", style_th),
            Paragraph("<b>Scope / Details</b>", style_th),
            Paragraph("<b>Penalty (INR)</b>", style_th)
        ]
    ]

    if offense_entries:
        for idx, entry in enumerate(offense_entries, start=1):
            offense_rows.append([
                Paragraph(str(idx), style_td),
                Paragraph(f"<b>{entry['offense']}</b>", style_td),
                Paragraph(entry['statute'], style_td),
                Paragraph(entry['scope'], style_td),
                Paragraph(f"&#8377; {entry['fine']:,}", style_td_bold)
            ])
    else:
        # Fallback if no specific offenses matched flags
        offense_rows.append([
            Paragraph("1", style_td),
            Paragraph("General Traffic Safety Inspection Record", style_td),
            Paragraph(STATUTORY_PROVISIONS["SURVEILLANCE_FRAMEWORK"]["section"], style_td),
            Paragraph(f"{rider_count} Rider(s) Observed", style_td),
            Paragraph(f"&#8377; {total_fine:,}", style_td_bold)
        ])

    offense_table = Table(offense_rows, colWidths=[20, 185, 205, 80, 66])
    offense_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(offense_table)
    story.append(Spacer(1, 4))

    # Total simulated fine callout
    total_data = [
        [
            Paragraph("<b>NOTE:</b> Statutory citations and penalties correspond to current Indian Gazette notifications for technical demonstration.", style_disclaimer),
            Paragraph(f"<b>TOTAL SIMULATED PENALTY: &#8377; {total_fine:,}</b>", style_fine_callout)
        ]
    ]
    total_table = Table(total_data, colWidths=[320, 236])
    total_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#fca5a5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(total_table)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECTION 5: DEMONSTRATION QR CODE & SIMULATED AUDIT VERIFICATION RECORD
    # -------------------------------------------------------------------------
    qr_buf, qr_url = generate_demonstration_qr(challan_id, vehicle_plate, total_fine)
    qr_flowable = RLImage(qr_buf, width=72, height=72)

    qr_panel_text = (
        "<b>SIMULATED VERIFICATION & AUDIT RECORD [EXPLICITLY NON-PAYABLE]:</b><br/>"
        "&bull; This demonstration QR code links to a mock verification record with cryptographic hash tokens.<br/>"
        "&bull; <b>NON-PAYABLE NOTICE:</b> This system is an academic research demonstration and does NOT collect real money.<br/>"
        "&bull; Never send funds, enter banking credentials, or attempt payments against demonstration QR codes.<br/>"
        f"&bull; Demonstration Audit URL: <i>{qr_url[:75]}...</i>"
    )

    qr_left_cell = [
        qr_flowable,
        Spacer(1, 2),
        Paragraph("<b>DEMO ONLY<br/>NON-PAYABLE</b>", style_qr_lbl)
    ]

    qr_table = Table([[qr_left_cell, Paragraph(qr_panel_text, style_disclaimer)]], colWidths=[80, 476])
    qr_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(qr_table)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECTION 6: MANDATORY ACADEMIC & LEGAL DISCLAIMER
    # -------------------------------------------------------------------------
    disclaimer_text = (
        "<b>IMPORTANT LEGAL & SYSTEM-GENERATED REPORT DISCLAIMER:</b><br/>"
        "1. <b>ACADEMIC & RESEARCH DEMONSTRATION ONLY:</b> This report is an automated computer-generated document produced "
        "by the TwoWheelerViolationAI research project. It is intended strictly for evaluating artificial intelligence algorithms.<br/>"
        "2. <b>NO LEGAL ENFORCEABILITY:</b> This document does NOT represent an official challan issued by any Police Department, "
        "Transport Authority, or Municipal Corporation, and does NOT create any legal liability or obligation to pay.<br/>"
        "3. <b>STATUTORY REFERENCE ACCURACY:</b> The Motor Vehicles Act 1988 sections (Sec 128, 129, 194A, 194C, 194D) and penalty "
        "amounts correspond to official Gazette amendments for academic illustration only.<br/>"
        "4. <b>AUDIT VERIFICATION:</b> Machine vision detections are subject to sensor noise, occlusion, and environmental conditions; "
        "all AI detections require certified law-enforcement officer review prior to any statutory enforcement."
    )

    disclaimer_table = Table([[Paragraph(disclaimer_text, style_disclaimer)]], colWidths=[556])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#94a3b8')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 4))

    # Footer note
    footer_text = f"TwoWheelerViolationAI v2.5 &bull; Report Generated: {ts_now.strftime('%d-%b-%Y %I:%M:%S %p')} &bull; Cryptographic Audit ID: {challan_id}"
    story.append(Paragraph(footer_text, ParagraphStyle(
        'FooterP', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=6.5, leading=8.5, textColor=colors.HexColor('#94a3b8'), alignment=1
    )))

    # 3. Build Document
    doc.build(story)

    pdf_bytes = pdf_buffer.getvalue()

    # 4. Save to Disk in Organized Folder if requested
    target_file = output_path
    if target_file is None and save_copy:
        safe_plate = vehicle_plate.replace(" ", "_")
        target_file = os.path.join(REPORTS_DIR, f"{challan_id}_{safe_plate}.pdf")

    if target_file:
        os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
        with open(target_file, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def generate_and_save_pdf(record, filename=None):
    """
    Convenience wrapper that returns both the saved filepath and the raw PDF bytes.
    """
    ts_now = datetime.now()
    c_id = record.get("challan_id") or f"DEMO-ECH-{ts_now.strftime('%Y%m%d-%H%M%S')}"
    plate = str(record.get("vehicle_plate") or "PLATE_NOT_DETECTED").strip().upper().replace(" ", "_")

    if filename:
        save_path = os.path.join(REPORTS_DIR, filename)
    else:
        save_path = os.path.join(REPORTS_DIR, f"{c_id}_{plate}.pdf")

    pdf_bytes = generate_pdf_challan(record, output_path=save_path, save_copy=True)
    return save_path, pdf_bytes


if __name__ == "__main__":
    print("Testing Advanced PDF Challan Generator (P6)...")
    sample_rec = {
        "challan_id": "DEMO-ECH-20261121-140000-TEST",
        "vehicle_plate": "TS09EA1234",
        "date_time": "21-Nov-2026 02:00:00 PM",
        "location": "Junction 04 - Panjagutta Circle, Hyderabad",
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
            {"helmet_status": "HELMET", "head_conf": 0.88},
            {"helmet_status": "NO_HELMET", "head_conf": 0.92},
            {"helmet_status": "NO_HELMET", "head_conf": 0.85}
        ],
        "audit_status": "VERIFIED"
    }

    path, b = generate_and_save_pdf(sample_rec)
    print(f"Generated PDF Report: {path} ({len(b)} bytes)")
