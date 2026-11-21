import cv2
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO

# Import project modules
from plate_ocr import extract_plate_details
from fine_generator import calculate_and_generate_challan
from head_association import associate_riders_with_helmets, annotate_individual_riders
from multimodal_verifier import verify_plate_with_gemini, compare_and_reconcile_plate


def filter_duplicate_riders(boxes, x_thresh_ratio=0.18, iou_thresh=0.60):
    """Suppresses duplicate bounding boxes for seated riders."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    kept = []

    for b in boxes:
        bx1, by1, bx2, by2 = b["box"]
        b_center_x = (bx1 + bx2) / 2.0
        b_width = bx2 - bx1
        b_area = b_width * (by2 - by1)

        is_duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["box"]
            k_center_x = (kx1 + kx2) / 2.0
            k_width = kx2 - kx1

            x_dist = abs(b_center_x - k_center_x)
            avg_width = (b_width + k_width) / 2.0

            ix1 = max(bx1, kx1)
            iy1 = max(by1, ky1)
            ix2 = min(bx2, kx2)
            iy2 = min(by2, ky2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih

            k_area = (kx2 - kx1) * (ky2 - ky1)
            union_area = b_area + k_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            if (x_dist < avg_width * x_thresh_ratio) or (iou > iou_thresh):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(b)

    return kept


def draw_label(image, text, x, y, bg_color, text_color=(255, 255, 255), scale=0.50, thickness=1):
    """Draws a badge label with a solid background rectangle."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (x, max(0, y - h - 6)), (x + w + 6, y + baseline), bg_color, -1)
    cv2.putText(image, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


class VideoTrafficAuditor:
    """
    Real-time / Batch Video Traffic Violation Processor and Multi-Object Tracker.
    Tracks motorcycles with ByteTrack, evaluates rider helmet and triple-riding violations,
    performs license plate extraction, and enforces intelligent cooldown deduplication.
    """

    def __init__(
        self,
        vehicle_model_path="yolov8n.pt",
        helmet_model_path="models/helmet_model.pt",
        cooldown_seconds=60,
        ocr_interval_frames=5,
        evidence_dir="evidence"
    ):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.helmet_model = YOLO(helmet_model_path)
        self.cooldown_seconds = cooldown_seconds
        self.ocr_interval_frames = ocr_interval_frames
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Vehicle tracking states: track_id -> dict
        self.tracked_vehicles = {}

        # Global Plate registry to prevent duplicate fines across ID switches:
        # plate_number -> last_challan_timestamp
        self.issued_plates = {}

        # Session metrics
        self.total_frames_processed = 0
        self.unique_vehicles_tracked = set()
        self.unique_violations_recorded = 0
        self.confirmed_challans_issued = 0

    def process_video(
        self,
        video_source,
        output_video_path=None,
        show_live=True,
        max_frames=None,
        enable_gemini_fallback=True
    ):
        """
        Processes a video file (MP4, AVI) or live RTSP stream.
        
        Controls during playback:
          'q' or ESC: Quit / Stop
          'p' or SPACE: Pause / Resume
          's': Save manual snapshot
        """
        is_stream = str(video_source).startswith(("rtsp://", "http://", "https://"))
        print("=" * 65)
        print("   TWO-WHEELER VIDEO VIOLATION PROCESSOR (P4)")
        print("=" * 65)
        print(f"Source       : {video_source}")
        print(f"Source Type  : {'Live Network Stream' if is_stream else 'Video File'}")
        print(f"Tracking     : ByteTrack (persist=True)")
        print(f"Deduplication: {self.cooldown_seconds}s cooldown per vehicle / plate")
        print("=" * 65)

        cap = cv2.VideoCapture(video_source)
        if is_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[Error] Unable to open video source: {video_source}")
            return self._build_session_summary()

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Saving annotated output video -> {output_video_path}")

        frame_idx = 0
        paused = False
        consecutive_read_failures = 0
        start_time = time.time()

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_stream and consecutive_read_failures < 5:
                        consecutive_read_failures += 1
                        time.sleep(0.5)
                        continue
                    else:
                        print("\nEnd of video stream reached or playback completed.")
                        break

                consecutive_read_failures = 0
                frame_idx += 1
                self.total_frames_processed += 1

                annotated_frame, _ = self._process_single_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    fps=fps,
                    enable_gemini=enable_gemini_fallback
                )

                if writer:
                    writer.write(annotated_frame)

            # Display window and controls
            if show_live:
                display_img = annotated_frame if not paused else frame
                cv2.imshow("Traffic Video Violation Monitoring (P4 - ByteTrack)", display_img)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF

                if key in [ord('q'), 27]:  # 'q' or ESC
                    print("\n[User Control] Stop command triggered by operator.")
                    break
                elif key in [ord('p'), 32]:  # 'p' or SPACE
                    paused = not paused
                    print(f"\n[User Control] Playback {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('s'):
                    snap_file = os.path.join(self.evidence_dir, f"manual_snap_frame_{frame_idx}.jpg")
                    cv2.imwrite(snap_file, display_img)
                    print(f"\n[Snapshot] Saved manual frame snapshot -> {snap_file}")

            if max_frames and frame_idx >= max_frames:
                print(f"\nTarget frame count ({max_frames}) reached.")
                break

        total_elapsed = max(0.001, time.time() - start_time)
        avg_fps = float(frame_idx) / total_elapsed

        cap.release()
        if writer:
            writer.release()
        if show_live:
            cv2.destroyAllWindows()

        summary = self._build_session_summary(avg_fps=avg_fps)
        self._print_session_report(summary)
        return summary

    def process_stream_generator(self, video_source, skip_frames=1, enable_gemini=False):
        """
        Stream generator for interactive Streamlit web dashboard.
        Yields: (frame_idx, annotated_frame_rgb, active_track_count, new_violations_list)
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[Error] Stream generator could not open: {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if skip_frames > 1 and (frame_idx % skip_frames != 0):
                    continue

                self.total_frames_processed += 1
                annotated_bgr, new_viols = self._process_single_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    fps=fps,
                    enable_gemini=enable_gemini
                )

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                active_count = len([v for v in self.tracked_vehicles.values() if v["last_seen_frame"] == frame_idx])
                yield frame_idx, annotated_rgb, active_count, new_viols
        finally:
            cap.release()

    def _process_single_frame(self, frame, frame_idx, fps, enable_gemini=True):
        """Processes detection, tracking, spatial association, and deduplication for a single frame."""
        annotated = frame.copy()
        current_time = time.time()
        newly_issued = []

        # 1. ByteTrack Object Tracking
        # Track persons and motorcycles
        track_results = self.vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )

        raw_persons = []
        motorcycles = []

        if track_results and len(track_results) > 0:
            boxes = track_results[0].boxes
            for b in boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                label = self.vehicle_model.names[cls]
                coords = list(map(int, b.xyxy[0]))
                track_id = int(b.id[0]) if b.id is not None else None

                if label == "person":
                    raw_persons.append({
                        "box": coords,
                        "conf": conf,
                        "track_id": track_id
                    })
                elif label == "motorcycle":
                    motorcycles.append({
                        "box": coords,
                        "conf": conf,
                        "track_id": track_id,
                        "riders": []
                    })

        persons = filter_duplicate_riders(raw_persons, x_thresh_ratio=0.18, iou_thresh=0.60)

        # 2. Spatial Association: Riders to Motorcycles
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

        # 3. Helmet Detection
        helmet_results = self.helmet_model(frame, conf=0.30, verbose=False)
        helmet_detections = []

        if helmet_results and len(helmet_results) > 0:
            for b in helmet_results[0].boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                label = self.helmet_model.names[cls].lower().strip()
                coords = list(map(int, b.xyxy[0]))
                is_viol = ("without" in label or "no" in label)

                helmet_detections.append({
                    "box": coords,
                    "conf": conf,
                    "label": label,
                    "is_violation": is_viol
                })

        # 4. Associate Riders to Head Zones and Update Tracking States
        for bike in motorcycles:
            track_id = bike.get("track_id")
            if track_id is None:
                continue

            self.unique_vehicles_tracked.add(track_id)

            # Associate riders with upper 30% head zones
            bike["riders"] = associate_riders_with_helmets(
                bike["riders"],
                helmet_detections,
                head_ratio=0.30
            )

            rider_count = len(bike["riders"])
            no_helmet_count = sum(1 for r in bike["riders"] if r["helmet_status"] == "NO_HELMET")
            is_triple_riding = (rider_count >= 3)
            has_violation = is_triple_riding or (no_helmet_count > 0)

            # Update vehicle tracking registry
            if track_id not in self.tracked_vehicles:
                self.tracked_vehicles[track_id] = {
                    "track_id": track_id,
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "first_seen_time": current_time,
                    "last_seen_time": current_time,
                    "frames_tracked": 0,
                    "plate_number": "PLATE_NOT_DETECTED",
                    "plate_conf": 0.0,
                    "plate_valid": False,
                    "plate_crop": None,
                    "has_triple_riding": False,
                    "has_helmet_violation": False,
                    "max_riders": 0,
                    "max_unhelmeted": 0,
                    "challan_issued": False,
                    "challan_id": None,
                    "last_challan_time": 0.0
                }

            v_state = self.tracked_vehicles[track_id]
            v_state["last_seen_frame"] = frame_idx
            v_state["last_seen_time"] = current_time
            v_state["frames_tracked"] += 1
            v_state["max_riders"] = max(v_state["max_riders"], rider_count)
            v_state["max_unhelmeted"] = max(v_state["max_unhelmeted"], no_helmet_count)

            if is_triple_riding:
                v_state["has_triple_riding"] = True
            if no_helmet_count > 0:
                v_state["has_helmet_violation"] = True

            # 5. Targeted Number Plate OCR (Run every N frames or when violation first occurs)
            should_run_ocr = (
                has_violation
                and (not v_state["plate_valid"])
                and (v_state["frames_tracked"] % self.ocr_interval_frames == 0 or v_state["frames_tracked"] == 1)
            )

            if should_run_ocr:
                try:
                    raw_txt, cand_plate, ocr_c, crop_img, is_valid = extract_plate_details(
                        frame,
                        motorcycle_box=bike["box"],
                        min_conf=0.10
                    )
                    if ocr_c > v_state["plate_conf"]:
                        v_state["plate_number"] = cand_plate
                        v_state["plate_conf"] = ocr_c
                        v_state["plate_valid"] = is_valid
                        v_state["plate_crop"] = crop_img
                except Exception as e:
                    pass

            # 6. Event-Deduplication & e-Challan Issuance
            # Check cooldown against both track_id and plate number
            time_since_last = current_time - v_state["last_challan_time"]
            is_in_cooldown = (v_state["challan_issued"] and time_since_last < self.cooldown_seconds)

            # Cross-plate check
            plate_key = v_state["plate_number"]
            if plate_key not in ["PLATE_NOT_DETECTED", "UNKNOWN"]:
                last_plate_time = self.issued_plates.get(plate_key, 0.0)
                if current_time - last_plate_time < self.cooldown_seconds:
                    is_in_cooldown = True

            # If violation is sustained (at least 2 frames) and not in cooldown:
            if has_violation and not is_in_cooldown and v_state["frames_tracked"] >= 2:
                rec = self._issue_deduplicated_violation(
                    vehicle_state=v_state,
                    evidence_frame=frame,
                    frame_idx=frame_idx,
                    current_time=current_time,
                    enable_gemini=enable_gemini
                )
                if rec:
                    newly_issued.append(rec)

        # 7. Render Visual Annotations
        self._annotate_video_frame(annotated, motorcycles, frame_idx, fps)
        return annotated, newly_issued

    def _issue_deduplicated_violation(self, vehicle_state, evidence_frame, frame_idx, current_time, enable_gemini=True):
        """Generates evidence snapshots and e-Challan with zero duplicates."""
        track_id = vehicle_state["track_id"]
        plate = vehicle_state["plate_number"]
        plate_valid = vehicle_state["plate_valid"]
        plate_conf = vehicle_state["plate_conf"]
        crop_img = vehicle_state["plate_crop"]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        evidence_filename = os.path.join(self.evidence_dir, f"violation_veh_{track_id}_frame_{frame_idx}_{timestamp_str}.jpg")
        cv2.imwrite(evidence_filename, evidence_frame)

        # Trigger Gemini Fallback if plate is uncertain and enabled
        final_plate = plate
        is_confirmed = plate_valid and (plate != "PLATE_NOT_DETECTED")
        audit_tag = "LOCAL_OCR_VALIDATED" if is_confirmed else "UNCERTAIN_LOCAL"

        if enable_gemini and (plate_conf < 0.50 or not plate_valid or plate == "PLATE_NOT_DETECTED"):
            gem_in = crop_img if (crop_img is not None and getattr(crop_img, "size", 0) > 0) else evidence_frame
            gem_res = verify_plate_with_gemini(gem_in)

            reconciled = compare_and_reconcile_plate(
                easyocr_result={"plate": plate, "conf": plate_conf, "is_valid": plate_valid, "raw": plate},
                gemini_result=gem_res
            )
            final_plate = reconciled["final_plate"]
            is_confirmed = reconciled["is_confirmed"]
            audit_tag = reconciled["verification_status"]

        self.unique_violations_recorded += 1
        vehicle_state["challan_issued"] = True
        vehicle_state["last_challan_time"] = current_time

        c_id = f"ECH-VID-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{track_id}"
        fine_amount = 0
        violations_list = []

        if is_confirmed and final_plate not in ["PLATE_NOT_DETECTED", "UNKNOWN"]:
            self.issued_plates[final_plate] = current_time
            challan_res = calculate_and_generate_challan(
                plate_number=final_plate,
                person_count=max(2, vehicle_state["max_riders"]),
                without_helmet_count=vehicle_state["max_unhelmeted"],
                output_image_path=evidence_filename,
                save_receipt=True
            )
            c_id = challan_res.get("challan_id", c_id)
            fine_amount = challan_res.get("total_fine", 0)
            violations_list = challan_res.get("violations", [])
            vehicle_state["challan_id"] = c_id
            self.confirmed_challans_issued += 1
            print(f"\n[EVENT LOG - FRAME {frame_idx}] Confirmed e-Challan issued for Vehicle #{track_id} (Plate: {final_plate})")
        else:
            if vehicle_state["has_triple_riding"]:
                violations_list.append(("Triple Riding on Two-Wheeler", 1000))
                fine_amount += 1000
            if vehicle_state["max_unhelmeted"] > 0:
                violations_list.append((f"Riding Without Helmet ({vehicle_state['max_unhelmeted']} Riders)", vehicle_state['max_unhelmeted'] * 1000))
                fine_amount += vehicle_state['max_unhelmeted'] * 1000
            vehicle_state["challan_id"] = c_id
            print(f"\n[EVENT LOG - FRAME {frame_idx}] Violation recorded for Vehicle #{track_id} | Status: PENDING_OFFICER_REVIEW (Plate: {final_plate})")

        rec = {
            "challan_id": c_id,
            "vehicle_plate": final_plate,
            "date_time": datetime.now().strftime("%d-%b-%Y %I:%M:%S %p"),
            "rider_count": max(1, vehicle_state["max_riders"]),
            "unhelmeted_count": vehicle_state["max_unhelmeted"],
            "triple_riding": vehicle_state["has_triple_riding"],
            "helmet_violation": vehicle_state["has_helmet_violation"],
            "total_fine": fine_amount,
            "violations": violations_list,
            "ocr_confidence": vehicle_state["plate_conf"],
            "plate_valid": is_confirmed,
            "evidence_path": evidence_filename,
            "audit_status": audit_tag,
            "source_type": "VIDEO"
        }

        try:
            import database
            database.insert_challan(rec)
        except Exception as dbe:
            print(f"[DB Warning] Error inserting video challan: {dbe}")

        return rec

    def _annotate_video_frame(self, frame, motorcycles, frame_idx, fps):
        """Draws tracked vehicle boxes, individual rider badges, and the live status HUD."""
        # 1. Draw Motorcycles
        for bike in motorcycles:
            bx1, by1, bx2, by2 = bike["box"]
            track_id = bike.get("track_id", "N/A")
            rc = len(bike["riders"])
            no_h = sum(1 for r in bike["riders"] if r.get("helmet_status") == "NO_HELMET")
            is_viol = (rc >= 3) or (no_h > 0)

            color = (0, 0, 255) if is_viol else (255, 120, 0)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            draw_label(
                frame,
                f"Motorcycle #{track_id} | Riders: {rc}",
                bx1 + 4,
                max(by1 - 10, 20),
                color,
                scale=0.52
            )

        # 2. Draw Individual Riders (P2)
        annotate_individual_riders(frame, motorcycles, show_head_zone=True)

        # 3. Live Video HUD Overlay
        cv2.rectangle(frame, (15, 15), (450, 140), (25, 25, 25), -1)
        cv2.rectangle(frame, (15, 15), (450, 140), (200, 200, 200), 2)

        cv2.putText(frame, "TRAFFIC VIOLATION MONITOR (P4)", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx:<6} | FPS: {fps:.1f}", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(frame, f"Tracked 2-Wheelers : {len(self.unique_vehicles_tracked)}", (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(frame, f"Unique Violations  : {self.unique_violations_recorded}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 100, 255), 2)
        cv2.putText(frame, f"Confirmed Challans : {self.confirmed_challans_issued}", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 2)

    def _build_session_summary(self, avg_fps=0.0):
        return {
            "total_frames_processed": self.total_frames_processed,
            "unique_vehicles_tracked": len(self.unique_vehicles_tracked),
            "unique_violations_recorded": self.unique_violations_recorded,
            "confirmed_challans_issued": self.confirmed_challans_issued,
            "avg_fps": round(avg_fps, 1),
            "tracked_vehicles": self.tracked_vehicles
        }

    def _print_session_report(self, summary):
        print("\n" + "=" * 65)
        print("          TRAFFIC VIDEO AUDIT SUMMARY REPORT (P4)")
        print("=" * 65)
        print(f"Total Frames Processed   : {summary['total_frames_processed']}")
        print(f"Average Processing Speed : {summary['avg_fps']} FPS")
        print(f"Unique Vehicles Tracked  : {summary['unique_vehicles_tracked']}")
        print(f"Unique Violations Logged : {summary['unique_violations_recorded']}")
        print(f"Confirmed Challans Issued: {summary['confirmed_challans_issued']}")
        print("-" * 65)
        print("Vehicle Tracking Log:")
        if not summary["tracked_vehicles"]:
            print("  (No two-wheelers tracked)")
        for t_id, v in summary["tracked_vehicles"].items():
            viol_str = []
            if v["has_triple_riding"]:
                viol_str.append("Triple Riding")
            if v["has_helmet_violation"]:
                viol_str.append(f"No Helmet ({v['max_unhelmeted']} riders)")
            status = "CONFIRMED CHALLAN" if v["challan_issued"] and v["challan_id"] else ("PENDING REVIEW" if viol_str else "NORMAL")
            print(f"  • Vehicle #{t_id:<2} | Plate: {v['plate_number']:<15} | Max Riders: {v['max_riders']} | Violations: {', '.join(viol_str) or 'None':<22} | Status: {status}")
        print("=" * 65)


def process_traffic_video(
    video_path,
    output_video_path=None,
    cooldown_seconds=60,
    show_live=True,
    max_frames=None,
    enable_gemini_fallback=True
):
    """Convenience function to execute video processing."""
    auditor = VideoTrafficAuditor(cooldown_seconds=cooldown_seconds)
    return auditor.process_video(
        video_source=video_path,
        output_video_path=output_video_path,
        show_live=show_live,
        max_frames=max_frames,
        enable_gemini_fallback=enable_gemini_fallback
    )
