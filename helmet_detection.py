from ultralytics import YOLO
import cv2
import os
import sys

# Import rider-to-head spatial association module (P2)
from head_association import associate_riders_with_helmets, annotate_individual_riders


# =========================================================
# HELPER: REMOVE DUPLICATE RIDER BOXES
# =========================================================

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


def run_helmet_audit(image_path="test.jpg.jpg", no_gui=False):
    """Executes the rider-to-head spatial association and helmet compliance audit."""
    print("=" * 60)
    print("   RIDER-TO-HEAD SPATIAL ASSOCIATION & HELMET AUDIT (P2)")
    print("=" * 60)

    vehicle_model_path = "yolov8n.pt"
    helmet_model_path = "models/helmet_model.pt"

    if not os.path.exists(image_path):
        for candidate in ["test.jpg.jpg", "test.jpg", "test_with_plate.jpg"]:
            if os.path.exists(candidate):
                image_path = candidate
                break

    if not os.path.exists(vehicle_model_path):
        print(f"ERROR: Vehicle model not found at {vehicle_model_path}")
        return

    if not os.path.exists(helmet_model_path):
        print(f"ERROR: Helmet model not found at {helmet_model_path}")
        return

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    print(f"Input image loaded : {image_path}")
    print("Loading YOLO models...")
    vehicle_model = YOLO(vehicle_model_path)
    helmet_model = YOLO(helmet_model_path)
    print("Models loaded successfully.\n")

    img = cv2.imread(image_path)
    results_vehicle = vehicle_model(image_path, conf=0.25, verbose=False)

    raw_persons = []
    motorcycles = []

    for r in results_vehicle:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            label = vehicle_model.names[cls]
            coords = list(map(int, b.xyxy[0]))

            if label == "person":
                raw_persons.append({"box": coords, "conf": conf})
            elif label == "motorcycle":
                motorcycles.append({"box": coords, "conf": conf, "riders": []})

    persons = filter_duplicate_riders(raw_persons, x_thresh_ratio=0.18, iou_thresh=0.60)

    for p in persons:
        px1, py1, px2, py2 = p["box"]
        pcx = (px1 + px2) / 2.0
        p_bottom = py2

        matched_bike = None
        max_overlap_x = 0

        for bike in motorcycles:
            bx1, by1, bx2, by2 = bike["box"]
            bw = bx2 - bx1

            is_x_aligned = (bx1 - 0.10 * bw) <= pcx <= (bx2 + 0.10 * bw)
            is_y_aligned = (p_bottom >= by1) and (py1 <= by2)

            if is_x_aligned and is_y_aligned:
                overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
                if overlap_x > max_overlap_x:
                    max_overlap_x = overlap_x
                    matched_bike = bike

        if matched_bike is not None:
            matched_bike["riders"].append(p)

    results_helmet = helmet_model(image_path, conf=0.30, verbose=False)
    helmet_detections = []

    for r in results_helmet:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            raw_label = helmet_model.names[cls]
            label = raw_label.lower().strip()
            coords = list(map(int, b.xyxy[0]))
            is_violation = ("without" in label or "no" in label)

            helmet_detections.append({
                "box": coords,
                "conf": conf,
                "label": label,
                "raw_label": raw_label,
                "is_violation": is_violation
            })

    total_with_helmet = 0
    total_without_helmet = 0
    total_unknown = 0
    global_rider_counter = 1

    for bike_idx, bike in enumerate(motorcycles, start=1):
        bike_riders = bike["riders"]
        associated_riders = associate_riders_with_helmets(
            riders=bike_riders,
            helmet_detections=helmet_detections,
            head_ratio=0.30
        )

        for r in associated_riders:
            r["rider_id"] = global_rider_counter
            global_rider_counter += 1

            status = r["helmet_status"]
            if status == "HELMET":
                total_with_helmet += 1
                r["status_label"] = "Helmet OK"
            elif status == "NO_HELMET":
                total_without_helmet += 1
                r["status_label"] = "No Helmet"
            else:
                total_unknown += 1
                r["status_label"] = "Unknown"

        bike["riders"] = associated_riders

    # Visual Annotations
    for bike in motorcycles:
        bx1, by1, bx2, by2 = bike["box"]
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 140, 0), 2)
        cv2.putText(
            img,
            f"Motorcycle: {len(bike['riders'])} Rider(s)",
            (bx1, max(by1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 140, 0),
            2
        )

    annotate_individual_riders(img, motorcycles, show_head_zone=True)

    output_path = "helmet_individual_output.jpg"
    cv2.imwrite(output_path, img)

    print("\n" + "=" * 55)
    print("       INDIVIDUAL RIDER HELMET REPORT (P2)")
    print("=" * 55)

    for i, bike in enumerate(motorcycles, start=1):
        print(f"Motorcycle {i}:")
        if not bike["riders"]:
            print("  (No riders associated)")
        for r in bike["riders"]:
            status_str = r["status_label"]
            conf_str = f" [conf: {r['helmet_conf']:.2f}]" if r["helmet_conf"] > 0 else ""
            print(f"- Rider {r['rider_id']}: {status_str}{conf_str}")

    print("-" * 55)
    print(f"Total Riders Audited     : {total_with_helmet + total_without_helmet + total_unknown}")
    print(f"Riders With Helmet       : {total_with_helmet}")
    print(f"Riders Without Helmet    : {total_without_helmet}")
    print(f"Uncertain / Unknown      : {total_unknown}")
    print("-" * 55)

    if total_without_helmet > 0:
        print("STATUS: [!] HELMET VIOLATION DETECTED")
    else:
        print("STATUS: [OK] NO CONFIRMED HELMET VIOLATION")

    print("=" * 55)
    print(f"Annotated Evidence Saved : {output_path}")

    if not no_gui:
        cv2.imshow("Individual Rider Helmet Detection (P2)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[GUI display skipped via --no-gui]")


if __name__ == "__main__":
    no_gui_flag = "--no-gui" in sys.argv
    img_arg = "test.jpg.jpg"
    for i, a in enumerate(sys.argv):
        if a == "--image" and i + 1 < len(sys.argv):
            img_arg = sys.argv[i + 1]
    run_helmet_audit(image_path=img_arg, no_gui=no_gui_flag)
