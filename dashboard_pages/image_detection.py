import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import json

# Project AI modules
from ultralytics import YOLO
from head_association import associate_riders_with_helmets, annotate_individual_riders
from plate_roi import detect_license_plate_roi
from plate_ocr import extract_plate_details
from multimodal_verifier import verify_plate_with_gemini, compare_and_reconcile_plate
from fine_generator import calculate_and_generate_challan
from pdf_challan_generator import generate_pdf_challan
import database


@st.cache_resource
def load_detection_models():
    """Caches YOLOv8 vehicle and helmet detection models in memory."""
    v_model = YOLO("yolov8n.pt")
    h_model = YOLO("models/helmet_model.pt")
    return v_model, h_model


def render_image_page():
    """Renders the AI Image Traffic Violation Analysis page."""
    st.markdown("### 📸 Image-Based Traffic Violation Surveillance")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 13px; margin-top: -10px;'>"
        "Upload a traffic frame or pick a benchmark preset to execute the full AI pipeline: "
        "YOLOv8 vehicle detection &bull; Upper 30% Head-Spatial Mapping &bull; Plate ROI & Regex OCR &bull; Gemini Multimodal Fallback."
        "</p>",
        unsafe_allow_html=True
    )

    # 1. Image Input Selection
    c1, c2 = st.columns([1, 1])
    with c1:
        source_mode = st.radio("Select Image Source:", ["Sample Benchmark Presets", "Upload Custom Image"], horizontal=True)

    input_image_path = None
    uploaded_file = None

    if source_mode == "Sample Benchmark Presets":
        preset = st.selectbox(
            "Select Sample Scene:",
            [
                ("Sample 1: Triple Riding with Plate (test_with_plate.jpg)", "test_with_plate.jpg"),
                ("Sample 2: Overloaded Motorcycle (test.jpg.jpg)", "test.jpg.jpg"),
                ("Sample 3: Multi-Vehicle Road Scene (images.jpg.jpg)", "images.jpg.jpg")
            ],
            format_func=lambda x: x[0]
        )
        input_image_path = preset[1]
    else:
        uploaded_file = st.file_uploader("Upload Two-Wheeler Scene Image (JPG / PNG):", type=["jpg", "jpeg", "png"])

    # Pipeline Settings in expander
    with st.expander("⚙️ Advanced Detection Parameters", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            yolo_conf = st.slider("Vehicle Detection Confidence:", 0.10, 0.80, 0.25, 0.05)
        with p2:
            helmet_conf = st.slider("Helmet Classifier Confidence:", 0.10, 0.80, 0.30, 0.05)
        with p3:
            enable_gemini = st.checkbox("Enable Gemini Multimodal OCR Fallback", value=True, help="Invokes Gemini vision if local EasyOCR confidence is < 0.50 or plate format is ambiguous.")

    # Load input image
    image_bgr = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_filename = uploaded_file.name
    elif input_image_path and os.path.exists(input_image_path):
        image_bgr = cv2.imread(input_image_path)
        input_filename = os.path.basename(input_image_path)

    if image_bgr is None:
        st.warning("Please upload an image or select a valid sample preset above.")
        return

    # Trigger AI Pipeline
    if st.button("⚡ Run Full AI Detection & Violation Analysis", type="primary", use_container_width=True):
        with st.spinner("Executing YOLOv8 vehicle detection, spatial head mapping, and plate OCR..."):
            v_model, h_model = load_detection_models()

            # Step 1: Detect motorcycles and persons
            v_results = v_model(image_bgr, conf=yolo_conf, verbose=False)
            boxes = v_results[0].boxes
            persons = []
            motorcycles = []

            for b in boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                label = v_model.names[cls]
                coords = list(map(int, b.xyxy[0]))

                if label == "person":
                    persons.append({"box": coords, "conf": conf})
                elif label == "motorcycle":
                    motorcycles.append({"box": coords, "conf": conf, "riders": []})

            # Spatial association: riders to motorcycles
            for p in persons:
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

            # Step 2: Detect helmets
            h_results = h_model(image_bgr, conf=helmet_conf, verbose=False)
            helmet_detections = []
            if h_results and len(h_results) > 0:
                for b in h_results[0].boxes:
                    cls = int(b.cls[0])
                    conf = float(b.conf[0])
                    label = h_model.names[cls].lower().strip()
                    coords = list(map(int, b.xyxy[0]))
                    is_viol = ("without" in label or "no" in label)
                    helmet_detections.append({
                        "box": coords,
                        "conf": conf,
                        "label": label,
                        "is_violation": is_viol
                    })

            # Associate each bike's riders with helmets using upper 30% head mapping
            primary_bike = motorcycles[0] if motorcycles else None
            if primary_bike:
                primary_bike["riders"] = associate_riders_with_helmets(
                    primary_bike["riders"],
                    helmet_detections,
                    head_ratio=0.30
                )
                rider_list = primary_bike["riders"]
            else:
                rider_list = associate_riders_with_helmets(persons, helmet_detections, head_ratio=0.30)

            # Step 3: License Plate Extraction & OCR (P1)
            primary_bike_box = primary_bike["box"] if primary_bike else None
            raw_text, corrected_plate, ocr_conf, plate_crop, is_valid_plate = extract_plate_details(
                image_bgr,
                motorcycle_box=primary_bike_box,
                min_conf=0.10
            )

            # Step 4: Gemini Multimodal Fallback (P3)
            final_plate = corrected_plate
            gemini_used = False
            gemini_result = None
            verification_status = "LOCAL_OCR_VALID" if is_valid_plate else "UNCERTAIN_LOCAL"

            if enable_gemini and (ocr_conf < 0.50 or not is_valid_plate or corrected_plate == "PLATE_NOT_DETECTED"):
                gem_input = plate_crop if (plate_crop is not None and getattr(plate_crop, "size", 0) > 0) else image_bgr
                gem_res = verify_plate_with_gemini(gem_input)
                gemini_used = True
                gemini_result = gem_res

                reconciled = compare_and_reconcile_plate(
                    easyocr_result={"plate": corrected_plate, "conf": ocr_conf, "is_valid": is_valid_plate, "raw": raw_text},
                    gemini_result=gem_res
                )
                final_plate = reconciled["final_plate"]
                verification_status = reconciled["verification_status"]

           
            # Step 5: Visual Annotation
            annotated_img = annotate_individual_riders(
                image_bgr,
                motorcycles,
                show_head_zone=True
            )

            # Draw Plate Bounding Box
            plate_roi_res = detect_license_plate_roi(
                image_bgr,
                motorcycle_box=primary_bike_box
            )


            # Compute Violations and Penalty
            rider_count = len(rider_list)
            unhelmeted_count = sum(1 for r in rider_list if r.get("helmet_status") == "NO_HELMET")
            triple_riding = (rider_count >= 3)
            helmet_viol = (unhelmeted_count > 0)

            total_fine = 0
            violations_detected = []
            if triple_riding:
                fine = 2000 if rider_count > 3 else 1000
                desc = "Dangerous Overloading (Sec 194A MV Act)" if rider_count > 3 else "Triple Riding (Sec 128/194C MV Act)"
                violations_detected.append((desc, fine))
                total_fine += fine
            if helmet_viol:
                fine = unhelmeted_count * 1000
                violations_detected.append((f"Riding Without Helmet ({unhelmeted_count} Rider{'s' if unhelmeted_count > 1 else ''}) - Sec 129/194D", fine))
                total_fine += fine

            # Save annotated image
            out_img_path = "final_violation_output.jpg"
            cv2.imwrite(out_img_path, annotated_img)

            # Save results in session_state for persistence across reruns
            st.session_state["image_analysis_result"] = {
                "annotated_path": out_img_path,
                "plate_crop": plate_crop,
                "raw_text": raw_text,
                "corrected_plate": corrected_plate,
                "final_plate": final_plate,
                "ocr_conf": ocr_conf,
                "is_valid_plate": is_valid_plate,
                "verification_status": verification_status,
                "gemini_used": gemini_used,
                "gemini_result": gemini_result,
                "rider_list": rider_list,
                "rider_count": rider_count,
                "unhelmeted_count": unhelmeted_count,
                "triple_riding": triple_riding,
                "helmet_viol": helmet_viol,
                "total_fine": total_fine,
                "violations": violations_detected,
                "timestamp": datetime.now().strftime("%d-%b-%Y %I:%M:%S %p")
            }

    # 2. Display Results
    if "image_analysis_result" in st.session_state:
        res = st.session_state["image_analysis_result"]
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.success("✅ Surveillance Analysis Completed Successfully!")

        col_img, col_metrics = st.columns([1.1, 0.9])

        with col_img:
            tabs = st.tabs(["🎯 Annotated Detections", "📷 Original Image"])
            with tabs[0]:
                if os.path.exists(res["annotated_path"]):
                    st.image(res["annotated_path"], caption="AI Annotated Detections (Rider Head Association & Number Plate ROI)", use_container_width=True)
            with tabs[1]:
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original Surveillance Input Frame", use_container_width=True)

        with col_metrics:
            # Card 1: License Plate Extraction
            st.markdown("""
            <div class="content-card">
                <div class="card-title">🔢 License Plate Intelligence (P1 + P3)</div>
            """, unsafe_allow_html=True)

            lp_col1, lp_col2 = st.columns([1, 1.2])
            with lp_col1:
                if res["plate_crop"] is not None and getattr(res["plate_crop"], "size", 0) > 0:
                    st.image(cv2.cvtColor(res["plate_crop"], cv2.COLOR_BGR2RGB), caption="Processed Plate ROI Crop", width=140)
                else:
                    st.caption("No Plate ROI Isolated")

            with lp_col2:
                final_plate_str = res["final_plate"]
                st.markdown(f"<div class='plate-display'>{final_plate_str}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 11px; color: #94a3b8; margin-top: 6px;'>Raw OCR: <code>{res['raw_text'] or 'None'}</code></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 11px; color: #94a3b8;'>Confidence: <b>{res['ocr_conf'] * 100:.1f}%</b></div>", unsafe_allow_html=True)

                if res["verification_status"] in ["VERIFIED", "LOCAL_OCR_VALID"]:
                    st.markdown('<span class="badge badge-ok">PLATE VERIFIED</span>', unsafe_allow_html=True)
                elif res["verification_status"] == "REVIEW_REQUIRED":
                    st.markdown('<span class="badge badge-warn">OFFICER REVIEW REQUIRED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-viol">UNRESOLVED / NOT DETECTED</span>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Card 2: Individual Rider Head Association (P2)
            st.markdown("""
            <div class="content-card">
                <div class="card-title">🪖 Individual Rider Spatial Status (P2)</div>
            """, unsafe_allow_html=True)

            st.markdown(f"<b>Total Riders Detected:</b> <code>{res['rider_count']}</code>", unsafe_allow_html=True)

            for idx, r in enumerate(res["rider_list"], start=1):
                status = r.get("helmet_status", "UNKNOWN")
                h_conf = r.get("head_conf", 0.0)
                if status == "HELMET":
                    badge_html = f'<span class="badge badge-ok">🟢 Rider {idx}: Helmet OK ({h_conf * 100:.1f}%)</span>'
                elif status == "NO_HELMET":
                    badge_html = f'<span class="badge badge-viol">🔴 Rider {idx}: No Helmet ({h_conf * 100:.1f}%)</span>'
                else:
                    badge_html = f'<span class="badge badge-warn">🟡 Rider {idx}: Status Unknown</span>'

                st.markdown(f"<div style='margin-bottom: 6px;'>{badge_html}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Card 3: Penalties & Challan Actions
            st.markdown("""
            <div class="content-card">
                <div class="card-title">⚖️ Offense Assessment & e-Challan Issuance</div>
            """, unsafe_allow_html=True)

            if res["violations"]:
                for v_name, v_cost in res["violations"]:
                    st.markdown(f"• <b>{v_name}</b> : <font color='#f43f5e'>₹ {v_cost:,}</font>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: #f43f5e; margin: 10px 0;'>TOTAL PENALTY: ₹ {res['total_fine']:,}</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #10b981; font-weight: 600;'>🟢 No Traffic Violations Recorded for this Scene.</p>", unsafe_allow_html=True)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Actions: Save to DB and Download PDF
            act_col1, act_col2 = st.columns(2)
            challan_id = f"DEMO-ECH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            with act_col1:
                if st.button("💾 Record e-Challan to Database", use_container_width=True):
                    record = {
                        "challan_id": challan_id,
                        "vehicle_plate": res["final_plate"],
                        "date_time": res["timestamp"],
                        "rider_count": res["rider_count"],
                        "unhelmeted_count": res["unhelmeted_count"],
                        "triple_riding": res["triple_riding"],
                        "helmet_violation": res["helmet_viol"],
                        "total_fine": res["total_fine"],
                        "violations": res["violations"],
                        "ocr_confidence": res["ocr_conf"],
                        "plate_valid": res["is_valid_plate"],
                        "evidence_path": res["annotated_path"],
                        "audit_status": "VERIFIED" if res["is_valid_plate"] else "REVIEW_REQUIRED",
                        "source_type": "IMAGE"
                    }
                    success = database.insert_challan(record)
                    if success:
                        st.toast(f"e-Challan #{challan_id} saved to database!", icon="✅")
                        st.success(f"Recorded in SQLite with ID: {challan_id}")
                    else:
                        st.warning("Record already exists or failed to save.")

            with act_col2:
                # PDF Generation
                pdf_record = {
                    "challan_id": challan_id,
                    "vehicle_plate": res["final_plate"],
                    "date_time": res["timestamp"],
                    "location": "Surveillance Sector #04 - Junction Road Cam",
                    "rider_count": res["rider_count"],
                    "unhelmeted_count": res["unhelmeted_count"],
                    "triple_riding": res["triple_riding"],
                    "helmet_violation": res["helmet_viol"],
                    "total_fine": res["total_fine"],
                    "violations": res["violations"],
                    "evidence_path": res["annotated_path"],
                    "plate_crop": res.get("plate_crop"),
                    "rider_list": res.get("rider_list", []),
                    "ocr_confidence": res["ocr_conf"],
                    "plate_valid": res["is_valid_plate"],
                    "audit_status": "VERIFIED" if res["is_valid_plate"] else "REVIEW_REQUIRED"
                }
                pdf_bytes = generate_pdf_challan(pdf_record)
                st.download_button(
                    label="📄 Download Demonstration PDF Challan",
                    data=pdf_bytes,
                    file_name=f"Demo_Challan_{res['final_plate']}_{challan_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            st.markdown("</div>", unsafe_allow_html=True)
