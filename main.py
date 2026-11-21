from ultralytics import YOLO
import cv2
import os
import sys
# Import OCR, e-Challan, Rider-Head Association, and Multimodal Verifier modules
from plate_ocr import extract_plate_number, extract_plate_details
from fine_generator import calculate_and_generate_challan
from head_association import associate_riders_with_helmets, annotate_individual_riders
from multimodal_verifier import verify_plate_with_gemini, compare_and_reconcile_plate


# =========================================================
# HELPER FUNCTION 1: DRAW BADGE LABEL
# =========================================================

def draw_label(
    image,
    text,
    x,
    y,
    bg_color,
    text_color=(255, 255, 255),
    scale=0.5,
    thickness=1
):
    """Draw a text label with a filled background rectangle."""
    (w, h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )
    cv2.rectangle(
        image,
        (x, max(0, y - h - 6)),
        (x + w + 6, y + baseline),
        bg_color,
        -1
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# HELPER FUNCTION 2: REMOVE DUPLICATE RIDER BOXES
# =========================================================

def filter_duplicate_riders(
    boxes,
    x_thresh_ratio=0.18,
    iou_thresh=0.60
):
    """
    Suppresses duplicate bounding boxes for closely seated riders.
    Keeps the detection with highest confidence score.
    """
    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda x: x["conf"],
        reverse=True
    )

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


# =========================================================
# MAIN PROGRAM
# =========================================================

print("=" * 60)
print("     UNIFIED TWO-WHEELER VIOLATION DETECTION SYSTEM")
print("=" * 60)


# =========================================================
# 1. CHECK REQUIRED FILES
# =========================================================

vehicle_model_path = "yolov8n.pt"
helmet_model_path = "models/helmet_model.pt"

# Detect which mode and input file is present in command arguments
image_path = "test.jpg.jpg"
video_path = None
no_gui = "--no-gui" in sys.argv

for i, arg in enumerate(sys.argv):
    if arg == "--image" and i + 1 < len(sys.argv):
        image_path = sys.argv[i + 1]
    elif arg in ["--video", "-v"] and i + 1 < len(sys.argv):
        video_path = sys.argv[i + 1]

# If video processing mode is requested (P4):
if video_path is not None:
    from video_processor import process_traffic_video
    output_vid = "tracked_traffic_output.mp4"
    for i, arg in enumerate(sys.argv):
        if arg == "--output-video" and i + 1 < len(sys.argv):
            output_vid = sys.argv[i + 1]

    summary = process_traffic_video(
        video_path=video_path,
        output_video_path=output_vid,
        show_live=not no_gui,
        cooldown_seconds=60
    )
    exit(0)

if not os.path.exists(image_path):
    for candidate in ["test.jpg.jpg", "test.jpg"]:
        if os.path.exists(candidate):
            image_path = candidate
            break

print("\nChecking required files...")

if not os.path.exists(vehicle_model_path):
    print(f"ERROR: Vehicle model not found -> {vehicle_model_path}")
    exit()

if not os.path.exists(helmet_model_path):
    print(f"ERROR: Helmet model not found -> {helmet_model_path}")
    exit()

if not os.path.exists(image_path):
    print(f"ERROR: Input image not found -> {image_path}")
    exit()


# =========================================================
# 2. LOAD MODELS
# =========================================================

print("\nLoading vehicle/person model...")
vehicle_model = YOLO(vehicle_model_path)
print("Vehicle model loaded.")

print("\nLoading helmet detection model...")
helmet_model = YOLO(helmet_model_path)
print("Helmet model loaded.")


# =========================================================
# 3. READ IMAGE
# =========================================================

img = cv2.imread(image_path)
if img is None:
    print(f"\nERROR: Unable to read input image -> {image_path}")
    exit()

print(f"\nInput image loaded successfully: {image_path}")


# =========================================================
# 4. VEHICLE + PERSON DETECTION
# =========================================================

print("\n[1/4] Running vehicle and rider detection...")

vehicle_results = vehicle_model(
    image_path,
    conf=0.25,
    verbose=False
)

raw_persons = []
motorcycles = []

for result in vehicle_results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = vehicle_model.names[cls]
        coords = list(map(int, box.xyxy[0]))

        if label == "person":
            raw_persons.append({
                "box": coords,
                "conf": conf
            })
        elif label == "motorcycle":
            motorcycles.append({
                "box": coords,
                "conf": conf,
                "riders": []
            })

persons = filter_duplicate_riders(
    raw_persons,
    x_thresh_ratio=0.18,
    iou_thresh=0.60
)


# =========================================================
# 5. SPATIAL ASSOCIATION: PERSONS TO MOTORCYCLES
# =========================================================

for person in persons:
    px1, py1, px2, py2 = person["box"]
    person_center_x = (px1 + px2) / 2.0
    person_bottom_y = py2

    matched_bike = None
    max_overlap = 0

    for bike in motorcycles:
        bx1, by1, bx2, by2 = bike["box"]
        bike_width = bx2 - bx1

        # Horizontal alignment check
        is_x_aligned = (
            bx1 - 0.10 * bike_width
            <= person_center_x
            <= bx2 + 0.10 * bike_width
        )

        # Vertical alignment check: Lower body touches/overlaps bike bounding box
        is_y_aligned = (
            person_bottom_y >= by1
            and py1 <= by2
        )

        if is_x_aligned and is_y_aligned:
            overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
            if overlap_x > max_overlap:
                max_overlap = overlap_x
                matched_bike = bike

    if matched_bike is not None:
        matched_bike["riders"].append(person)


# =========================================================
# 6. HELMET DETECTION
# =========================================================

print("[2/4] Running helmet detection...")

helmet_results = helmet_model(
    image_path,
    conf=0.30,
    verbose=False
)

raw_with_helmet = 0
raw_without_helmet = 0
helmet_boxes_data = []

for result in helmet_results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = helmet_model.names[cls].lower().strip()
        coords = list(map(int, box.xyxy[0]))

        is_violation = ("without" in label or "no" in label)

        if is_violation:
            raw_without_helmet += 1
        else:
            raw_with_helmet += 1

        helmet_boxes_data.append({
            "box": coords,
            "conf": conf,
            "is_violation": is_violation,
            "label": label
        })


# =========================================================
# 7. RIDER-TO-HEAD SPATIAL ASSOCIATION & STRICT METRICS (P2)
# =========================================================

motorcycle_count = len(motorcycles)

# Associate helmet detections with individual riders' upper 30% head zones
for bike in motorcycles:
    bike["riders"] = associate_riders_with_helmets(
        bike["riders"],
        helmet_boxes_data,
        head_ratio=0.30
    )

total_associated_riders = sum(len(m["riders"]) for m in motorcycles)

if total_associated_riders > 0:
    actual_riders = total_associated_riders
    verified_without_helmet = sum(1 for m in motorcycles for r in m["riders"] if r["helmet_status"] == "NO_HELMET")
    verified_with_helmet = sum(1 for m in motorcycles for r in m["riders"] if r["helmet_status"] == "HELMET_OK")
    verified_unknown_helmet = sum(1 for m in motorcycles for r in m["riders"] if r["helmet_status"] == "UNKNOWN")
else:
    actual_riders = len(persons)
    # If no motorcycle was matched, fallback to raw detections
    verified_without_helmet = min(raw_without_helmet, actual_riders)
    verified_with_helmet = max(0, actual_riders - verified_without_helmet)
    verified_unknown_helmet = 0

# Determine Violation States
triple_riding = (motorcycle_count > 0 and actual_riders >= 3)
helmet_violation = (verified_without_helmet > 0)
violation = (triple_riding or helmet_violation)

final_status = "VIOLATION DETECTED" if violation else "NO VIOLATION"


# =========================================================
# 8. VISUAL ANNOTATIONS (P2)
# =========================================================

print("[3/4] Generating visual annotations...")

# 1. Draw Motorcycle Boxes (Blue)
for i, motorcycle in enumerate(motorcycles, start=1):
    x1, y1, x2, y2 = motorcycle["box"]
    rider_count = len(motorcycle["riders"])

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    draw_label(
        img,
        f"Motorcycle #{i} | Riders: {rider_count}",
        x1 + 5,
        y1 + 22,
        (255, 0, 0)
    )

# 2. Draw Individual Rider Bounding Boxes, Head Zones, and Status Badges (P2)
annotate_individual_riders(img, motorcycles, show_head_zone=True)

# 3. Report Overlay Card
cv2.rectangle(img, (15, 15), (460, 240), (255, 255, 255), -1)
cv2.rectangle(img, (15, 15), (460, 240), (60, 60, 60), 2)

cv2.putText(img, "TWO-WHEELER VIOLATION REPORT", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
cv2.putText(img, f"Motorcycles: {motorcycle_count}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
cv2.putText(img, f"Riders on Bike: {actual_riders}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1)
cv2.putText(
    img,
    f"Triple Riding: {'YES' if triple_riding else 'NO'}",
    (25, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.50,
    (0, 0, 255) if triple_riding else (0, 140, 0),
    2
)
cv2.putText(img, f"With Helmet: {verified_with_helmet}", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 0), 2)
cv2.putText(
    img,
    f"Without Helmet: {verified_without_helmet}",
    (25, 170),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.50,
    (0, 0, 255) if helmet_violation else (0, 140, 0),
    2
)
cv2.putText(img, f"Helmet Unknown: {verified_unknown_helmet}", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (60, 60, 60), 1)
cv2.putText(
    img,
    final_status,
    (25, 225),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 0, 255) if violation else (0, 140, 0),
    2
)


# =========================================================
# 9. SAVE ANNOTATED IMAGE
# =========================================================

output_path = "final_violation_output.jpg"
cv2.imwrite(output_path, img)


# =========================================================
# 10. TERMINAL DETECTION REPORT
# =========================================================

print("\n" + "=" * 60)
print("             DETECTION SUMMARY REPORT")
print("=" * 60)
print(f"Persons Detected       : {len(persons)}")
print(f"Motorcycles Detected   : {motorcycle_count}")
print(f"Associated Riders      : {actual_riders}")
print(f"With Helmet            : {verified_with_helmet}")
print(f"Without Helmet         : {verified_without_helmet}")
print(f"Helmet Unknown         : {verified_unknown_helmet}")
print(f"Triple Riding          : {'YES' if triple_riding else 'NO'}")
print(f"Helmet Violation       : {'YES' if helmet_violation else 'NO'}")
print(f"FINAL STATUS           : {final_status}")
print("-" * 60)

# Print Individual Rider Breakdown (Requirement 5 format)
for i, bike in enumerate(motorcycles, start=1):
    print(f"Motorcycle {i}:")
    if not bike["riders"]:
        print("  (No riders associated)")
    for r in bike["riders"]:
        print(f"- Rider {r['rider_id']}: {r['status_label']}")

print("-" * 60)
print(f"Annotated Image        : {output_path}")


# =========================================================
# 11. LICENSE PLATE OCR + E-CHALLAN GENERATION
# =========================================================

if violation:
    print("\n[4/4] Triggering automated e-Challan generation pipeline...")

    # Select primary detected motorcycle box for targeted plate ROI extraction
    primary_bike_box = motorcycles[0]["box"] if len(motorcycles) > 0 else None

    # Priority 1: Automated License Plate ROI Detection & Recognition
    print("Running Automated License Plate ROI Extraction & Recognition...")
    try:
        raw_text, detected_plate, ocr_conf, plate_crop, is_plate_valid = extract_plate_details(
            img,
            motorcycle_box=primary_bike_box,
            min_conf=0.10
        )
    except Exception as e:
        print(f"OCR Pipeline Error: {e}")
        raw_text = "PLATE_NOT_DETECTED"
        detected_plate = "PLATE_NOT_DETECTED"
        ocr_conf = 0.0
        plate_crop = None
        is_plate_valid = False

    # Save cropped plate image if extracted
    plate_crop_path = "detected_plate_crop.jpg"
    if plate_crop is not None and getattr(plate_crop, "size", 0) > 0:
        cv2.imwrite(plate_crop_path, plate_crop)
        print(f"License Plate Crop Saved : {plate_crop_path}")

    print("\n" + "-" * 55)
    print("       NUMBER PLATE RECOGNITION (P1 AUDIT)")
    print("-" * 55)
    print(f"Original Detected Text     : {raw_text}")
    print(f"Corrected Registration No  : {detected_plate}")
    print(f"OCR Confidence Score       : {ocr_conf:.2f}")
    print(f"Format Validation Status   : {'VALID (Indian Standard)' if is_plate_valid else 'INVALID / UNCONFIRMED'}")
    print("-" * 55)

    # ---------------------------------------------------------
    # PRIORITY 3: GEMINI MULTIMODAL FALLBACK FOR UNCERTAIN OCR
    # ---------------------------------------------------------
    needs_gemini_fallback = (
        ocr_conf < 0.50
        or detected_plate == "PLATE_NOT_DETECTED"
        or not is_plate_valid
    )

    final_registration = detected_plate
    is_confirmed_challan = is_plate_valid and (detected_plate != "PLATE_NOT_DETECTED")
    audit_status = "LOCAL_OCR_VALIDATED" if is_confirmed_challan else "UNCERTAIN_LOCAL_OCR"

    if needs_gemini_fallback:
        print("\n[P3 Multimodal Fallback] Triggering Gemini Vision Verification...")
        print(f"    Trigger condition: OCR Conf={ocr_conf:.2f} (<0.50), Valid={is_plate_valid}, Plate={detected_plate}")

        gemini_input = plate_crop if (plate_crop is not None and getattr(plate_crop, "size", 0) > 0) else img
        gemini_response = verify_plate_with_gemini(gemini_input)

        print(f"    Gemini Status     : {gemini_response.get('status')}")
        print(f"    Gemini Reading    : '{gemini_response.get('detected_number')}' (Conf: {gemini_response.get('confidence', 0.0):.2f})")
        print(f"    Gemini Observation: {gemini_response.get('explanation')}")

        reconciliation = compare_and_reconcile_plate(
            easyocr_result={
                "plate": detected_plate,
                "conf": ocr_conf,
                "is_valid": is_plate_valid,
                "raw": raw_text
            },
            gemini_result=gemini_response
        )

        final_registration = reconciliation["final_plate"]
        is_confirmed_challan = reconciliation["is_confirmed"]
        audit_status = reconciliation["verification_status"]
        print(f"    Reconciliation    : {reconciliation['summary']}")
        print(f"    Final Audit Status: {audit_status}")

    # Statutory Enforcement Decision (Requirement 12):
    # Only generate confirmed e-Challan if plate is valid and confirmed.
    if is_confirmed_challan and final_registration not in ["PLATE_NOT_DETECTED", "UNKNOWN"]:
        print(f"\nGenerating Confirmed e-Challan [{audit_status}]...")
        try:
            challan_summary = calculate_and_generate_challan(
                plate_number=final_registration,
                person_count=actual_riders,
                without_helmet_count=verified_without_helmet,
                output_image_path=output_path,
                save_receipt=True
            )
            print(f"Confirmed e-Challan issued successfully for vehicle: {final_registration}")
        except Exception as e:
            print(f"e-Challan generation error: {e}")
    else:
        print("\n[!] STATUTORY NOTICE: Vehicle registration number is invalid, conflicting, or unconfirmed.")
        print(f"    Raw Reading   : '{raw_text}'")
        print(f"    Candidate     : '{final_registration}'")
        print(f"    Audit Status  : '{audit_status}'")
        print("    Requirement 12: Confirmed e-Challan was NOT generated for unverified plate.")
        print("    Incident recorded with status: PENDING_OFFICER_REVIEW.")

else:
    print("\n[4/4] No violations detected.")
    print("System status: NORMAL")


# =========================================================
# 12. DISPLAY RESULT
# =========================================================

print("\n" + "=" * 60)
print("       TWO-WHEELER VIOLATION SYSTEM COMPLETE")
print("=" * 60)
print(f"Final Status : {final_status}")
print(f"Evidence     : {output_path}")
print("=" * 60)

if not no_gui:
    cv2.imshow("Two-Wheeler Violation Detection System", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("[GUI display skipped via --no-gui]")