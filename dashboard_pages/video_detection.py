import streamlit as st
import cv2
import os
import time
import tempfile
from datetime import datetime
from PIL import Image

from video_processor import VideoTrafficAuditor
from pdf_challan_generator import generate_pdf_challan
import database


def render_video_page():
    """Renders the AI Video Surveillance & Multi-Object Tracking page."""
    st.markdown("### 🎥 Real-Time Video Surveillance & ByteTrack Multi-Object Tracking")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 13px; margin-top: -10px;'>"
        "Process traffic video feeds (MP4/AVI) or live CCTV RTSP streams with persistent ByteTrack vehicle IDs, "
        "rider helmet evaluation, license plate OCR, and 60-second cooldown deduplication."
        "</p>",
        unsafe_allow_html=True
    )

    # 1. Video Source Selection
    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        source_mode = st.radio(
            "Select Video Source:",
            ["Benchmark Sample Videos", "Upload Video (MP4/AVI)", "Live RTSP Camera Stream"],
            horizontal=True
        )

    target_video_source = None

    if source_mode == "Benchmark Sample Videos":
        vid_choice = st.selectbox(
            "Select Surveillance Video Clip:",
            [
                ("Benchmark 1: High Density Motorcycle Video (test_traffic.mp4)", "test_traffic.mp4"),
                ("Benchmark 2: Tracked Traffic Output Clip (tracked_traffic_output.mp4)", "tracked_traffic_output.mp4")
            ],
            format_func=lambda x: x[0]
        )
        target_video_source = vid_choice[1]
    elif source_mode == "Upload Video (MP4/AVI)":
        uploaded_vid = st.file_uploader("Upload Traffic Surveillance Video:", type=["mp4", "avi", "mov"])
        if uploaded_vid is not None:
            # Save to temporary file for OpenCV VideoCapture
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.flush()
            target_video_source = tfile.name
    else:
        rtsp_url = st.text_input("Enter RTSP CCTV Stream URL:", placeholder="rtsp://admin:pass@192.168.1.100:554/stream1")
        if rtsp_url:
            target_video_source = rtsp_url

    # Settings in expander
    with st.expander("⚙️ Tracking & Processing Configuration", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            frame_skip = st.slider("Frame Skip Rate (1 = Full, 2 = 2x Speed):", 1, 4, 1)
        with c2:
            cooldown_sec = st.slider("Deduplication Cooldown (Seconds):", 10, 120, 60, 5)
        with c3:
            enable_gemini = st.checkbox("Gemini OCR Fallback on Uncertainty", value=False, help="Recommended off for high-FPS video streaming unless debugging.")

    if not target_video_source or not os.path.exists(str(target_video_source)):
        st.info("Please select or upload a valid video source to start the surveillance engine.")
        return

    # Session storage for live captured violations during this run
    if "video_captured_violations" not in st.session_state:
        st.session_state["video_captured_violations"] = []

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        start_btn = st.button("▶️ Start Video Surveillance", type="primary", use_container_width=True)
    with ctrl_col2:
        stop_btn = st.button("⏹️ Stop Surveillance", use_container_width=True)

    if stop_btn:
        st.session_state["stop_surveillance"] = True
        st.info("Stopping surveillance stream...")

    # Real-time Video Canvas & Telemetry Display
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    hud_c1, hud_c2, hud_c3, hud_c4 = st.columns(4)
    with hud_c1:
        metric_frames = st.empty()
        metric_frames.metric("Frames Processed", "0")
    with hud_c2:
        metric_vehicles = st.empty()
        metric_vehicles.metric("Active Tracked Vehicles", "0")
    with hud_c3:
        metric_violations = st.empty()
        metric_violations.metric("Deduplicated Violations", "0")
    with hud_c4:
        metric_fps = st.empty()
        metric_fps.metric("Processing Rate", "-- FPS")

    video_placeholder = st.empty()
    evidence_container = st.container()

    if start_btn:
        st.session_state["stop_surveillance"] = False
        st.session_state["video_captured_violations"] = []

        auditor = VideoTrafficAuditor(
            vehicle_model_path="yolov8n.pt",
            helmet_model_path="models/helmet_model.pt",
            cooldown_seconds=cooldown_sec,
            ocr_interval_frames=4,
            evidence_dir="evidence"
        )

        st.toast("Initializing ByteTrack tracker and surveillance engine...", icon="🚀")

        start_time = time.time()
        last_metric_update = time.time()
        frames_count = 0

        # Execute stream generator
        for frame_idx, frame_rgb, active_tracks, new_viols in auditor.process_stream_generator(
            video_source=target_video_source,
            skip_frames=frame_skip,
            enable_gemini=enable_gemini
        ):
            frames_count += 1

            # Check if user clicked Stop
            if st.session_state.get("stop_surveillance", False):
                break

            # Display annotated video frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Record newly captured violations
            if new_viols:
                for v in new_viols:
                    st.session_state["video_captured_violations"].append(v)
                    st.toast(f"🚨 Violation Detected for Vehicle (Plate: {v['vehicle_plate']})!", icon="🚨")

            # Update telemetry HUD every ~300ms to preserve UI responsiveness
            curr_time = time.time()
            if curr_time - last_metric_update > 0.30:
                elapsed = max(0.001, curr_time - start_time)
                current_fps = float(frames_count) / elapsed

                metric_frames.metric("Frames Processed", f"{frame_idx}")
                metric_vehicles.metric("Active Tracked Vehicles", f"{active_tracks}")
                metric_violations.metric("Deduplicated Violations", f"{auditor.unique_violations_recorded}")
                metric_fps.metric("Processing Rate", f"{current_fps:.1f} FPS")
                last_metric_update = curr_time

        st.success(
            f"Surveillance session finished! Processed {frames_count} frames. "
            f"Total unique vehicles tracked: {len(auditor.unique_vehicles_tracked)}. "
            f"Total confirmed violations: {auditor.unique_violations_recorded}."
        )

    # 3. Live Evidence Snapshot Wall
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📸 Captured Violation Evidence Snapshots (Session)")

    session_viols = st.session_state.get("video_captured_violations", [])
    if not session_viols:
        # If no new violations captured in this active click, show recent from database with source_type='VIDEO'
        recent_video_records = [c for c in database.get_all_challans() if c.get("source_type") == "VIDEO"][:4]
        if recent_video_records:
            st.caption("Displaying recent video surveillance evidence saved to database:")
            cols = st.columns(min(4, len(recent_video_records)))
            for i, rec in enumerate(recent_video_records):
                with cols[i]:
                    ev_path = rec.get("evidence_path", "")
                    if ev_path and os.path.exists(ev_path):
                        st.image(ev_path, caption=f"Plate: {rec['vehicle_plate']}", use_container_width=True)
                    st.markdown(f"<b>Challan:</b> <code>{rec['challan_id']}</code>", unsafe_allow_html=True)
                    st.markdown(f"<b>Fine:</b> <font color='#f43f5e'>₹ {rec['total_fine']:,}</font>", unsafe_allow_html=True)
                    st.markdown(f"<span class='badge badge-ok'>{rec['audit_status']}</span>", unsafe_allow_html=True)
        else:
            st.info("No video violation snapshots captured yet. Start video surveillance above to capture evidence.")
    else:
        grid_cols = st.columns(min(4, len(session_viols)))
        for idx, viol in enumerate(session_viols):
            col_idx = idx % min(4, len(session_viols))
            with grid_cols[col_idx]:
                ev_path = viol.get("evidence_path", "")
                if ev_path and os.path.exists(ev_path):
                    st.image(ev_path, caption=f"Vehicle Plate: {viol['vehicle_plate']}", use_container_width=True)
                st.markdown(f"<b>Challan ID:</b> <code>{viol['challan_id']}</code>", unsafe_allow_html=True)
                st.markdown(f"<b>Fine:</b> <font color='#f43f5e'>₹ {viol['total_fine']:,}</font>", unsafe_allow_html=True)
                st.markdown(f"<span class='badge badge-ok'>{viol['audit_status']}</span>", unsafe_allow_html=True)
