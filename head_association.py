import cv2
import numpy as np


def extract_head_box(rider_box, head_ratio=0.30):
    """
    Extracts the upper approximate head region of a rider bounding box.
    
    Args:
        rider_box (list or tuple): [x1, y1, x2, y2]
        head_ratio (float): Fraction of upper body to consider as head zone (default 0.30).
        
    Returns:
        list: [head_x1, head_y1, head_x2, head_y2]
    """
    rx1, ry1, rx2, ry2 = map(int, rider_box)
    rh = ry2 - ry1
    head_y2 = int(ry1 + max(head_ratio * rh, 15))
    return [rx1, ry1, rx2, head_y2]


def compute_spatial_overlap(head_box, helmet_box):
    """
    Computes spatial overlap and distance metrics between a rider's head zone
    and a detected helmet bounding box.
    """
    hx1, hy1, hx2, hy2 = head_box
    bx1, by1, bx2, by2 = helmet_box

    # Intersection box
    ix1 = max(hx1, bx1)
    iy1 = max(hy1, by1)
    ix2 = min(hx2, bx2)
    iy2 = min(hy2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = float(iw * ih)

    head_area = float(max(1, (hx2 - hx1) * (hy2 - hy1)))
    helmet_area = float(max(1, (bx2 - bx1) * (by2 - by1)))

    union_area = head_area + helmet_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    # Coverage: Fraction of the helmet box that lies within the rider head zone
    helmet_coverage = inter_area / helmet_area

    # Center distance normalized by head dimensions
    head_cx = (hx1 + hx2) / 2.0
    head_cy = (hy1 + hy2) / 2.0
    helmet_cx = (bx1 + bx2) / 2.0
    helmet_cy = (by1 + by2) / 2.0

    head_w = max(1.0, float(hx2 - hx1))
    head_h = max(1.0, float(hy2 - hy1))

    dist_x = abs(head_cx - helmet_cx) / head_w
    dist_y = abs(head_cy - helmet_cy) / head_h

    return {
        "iou": iou,
        "helmet_coverage": helmet_coverage,
        "dist_x": dist_x,
        "dist_y": dist_y,
        "inter_area": inter_area
    }


def associate_riders_with_helmets(riders, helmet_detections, head_ratio=0.30):
    """
    Associates helmet / no-helmet detections to individual seated riders
    using upper 30% head-zone spatial proximity and bounding box overlap.
    
    Args:
        riders (list of dict): List of rider dicts with 'box' [x1, y1, x2, y2] and 'conf'.
        helmet_detections (list of dict): List of helmet dicts with 'box', 'conf', 'label', 'is_violation'.
        head_ratio (float): Ratio of upper body considered head region.
        
    Returns:
        list of dict: Enriched rider dicts with:
            - 'rider_id': Temporary sequential ID (1, 2, ...)
            - 'head_box': [hx1, hy1, hx2, hy2]
            - 'helmet_status': 'HELMET_OK' | 'NO_HELMET' | 'UNKNOWN'
            - 'status_label': 'Helmet OK' | 'No Helmet' | 'Helmet Unknown'
            - 'helmet_conf': Confidence of matched helmet detection (or 0.0)
            - 'matched_helmet': Matched detection dict or None
    """
    if not riders:
        return []

    # Sort riders spatially: front to back along motorcycle
    # (By default x-center, or y-bottom for perspective)
    sorted_riders = sorted(
        riders,
        key=lambda r: (r["box"][0] + r["box"][2]) / 2.0
    )

    # Initialize rider structures
    associated_riders = []
    for idx, r in enumerate(sorted_riders, start=1):
        h_box = extract_head_box(r["box"], head_ratio=head_ratio)
        r_copy = dict(r)
        r_copy["rider_id"] = idx
        r_copy["head_box"] = h_box
        r_copy["helmet_status"] = "UNKNOWN"
        r_copy["status_label"] = "Helmet Unknown"
        r_copy["helmet_conf"] = 0.0
        r_copy["matched_helmet"] = None
        associated_riders.append(r_copy)

    if not helmet_detections:
        return associated_riders

    # Compute pairwise match scores between rider head boxes and helmet detections
    # Candidate matches: (score, rider_idx, helmet_idx)
    candidate_matches = []

    for r_idx, r in enumerate(associated_riders):
        head_box = r["head_box"]
        for h_idx, h in enumerate(helmet_detections):
            metrics = compute_spatial_overlap(head_box, h["box"])
            
            # Acceptance criteria:
            # 1. At least 20% of helmet lies inside head zone OR IoU >= 0.12
            # 2. OR centers are closely aligned (dist_x <= 0.45 and dist_y <= 0.75)
            is_valid_candidate = (
                metrics["helmet_coverage"] >= 0.20
                or metrics["iou"] >= 0.12
                or (metrics["dist_x"] <= 0.45 and metrics["dist_y"] <= 0.75 and metrics["inter_area"] > 0)
            )

            if is_valid_candidate:
                # Score combines coverage, IoU, and center proximity
                score = (
                    metrics["helmet_coverage"] * 2.0
                    + metrics["iou"] * 3.0
                    - (metrics["dist_x"] + metrics["dist_y"]) * 0.5
                    + h["conf"] * 0.2
                )
                candidate_matches.append((score, r_idx, h_idx))

    # Greedy optimal assignment: assign highest scoring pairs first
    candidate_matches.sort(key=lambda m: m[0], reverse=True)

    assigned_riders = set()
    assigned_helmets = set()

    for score, r_idx, h_idx in candidate_matches:
        if r_idx in assigned_riders or h_idx in assigned_helmets:
            continue

        rider = associated_riders[r_idx]
        helmet = helmet_detections[h_idx]

        rider["matched_helmet"] = helmet
        rider["helmet_conf"] = helmet["conf"]

        if helmet["is_violation"]:
            rider["helmet_status"] = "NO_HELMET"
            rider["status_label"] = "No Helmet"
        else:
            rider["helmet_status"] = "HELMET_OK"
            rider["status_label"] = "Helmet OK"

        assigned_riders.add(r_idx)
        assigned_helmets.add(h_idx)

    # Any rider with no matched helmet remains 'UNKNOWN' (Requirement 10:
    # Do not classify as helmetless merely because detector missed a detection)
    return associated_riders


def draw_label_badge(image, text, x, y, bg_color, text_color=(255, 255, 255), scale=0.48, thickness=1):
    """Draws a label with a solid badge background."""
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y_top = max(0, y - h - 6)
    cv2.rectangle(image, (x, y_top), (x + w + 8, y + baseline + 2), bg_color, -1)
    cv2.putText(image, text, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


def annotate_individual_riders(image, motorcycles, show_head_zone=True):
    """
    Draws individual rider bounding boxes, head zones, temporary rider IDs,
    and individual helmet statuses using color-coded labels:
      - Green: Helmet OK
      - Red: No Helmet
      - Amber/Orange: Helmet Unknown
    """
    for bike_idx, bike in enumerate(motorcycles, start=1):
        riders = bike.get("riders", [])

        for r in riders:
            rx1, ry1, rx2, ry2 = r["box"]
            rider_id = r.get("rider_id", 1)
            helmet_status = r.get("helmet_status", "UNKNOWN")
            status_label = r.get("status_label", "Helmet Unknown")
            conf = r.get("helmet_conf", 0.0)

            # Determine color by helmet status
            if helmet_status == "HELMET_OK":
                color = (0, 180, 0)      # Green
                tag = f"Rider #{rider_id}: {status_label} ({conf:.2f})" if conf > 0 else f"Rider #{rider_id}: {status_label}"
            elif helmet_status == "NO_HELMET":
                color = (0, 0, 220)      # Red
                tag = f"Rider #{rider_id}: {status_label} ({conf:.2f})" if conf > 0 else f"Rider #{rider_id}: {status_label}"
            else:
                color = (0, 160, 240)    # Amber / Orange for UNKNOWN
                tag = f"Rider #{rider_id}: {status_label}"

            # 1. Draw rider body box
            cv2.rectangle(image, (rx1, ry1), (rx2, ry2), color, 2)

            # 2. Draw upper 30% head zone bracket if requested
            if show_head_zone and "head_box" in r:
                hx1, hy1, hx2, hy2 = r["head_box"]
                # Subtle head-zone indicator
                cv2.rectangle(image, (hx1, hy1), (hx2, hy2), color, 1, lineType=cv2.LINE_AA)

            # 3. Draw ID and helmet status badge above rider
            badge_y = max(ry1 - 8, 22)
            draw_label_badge(image, tag, rx1, badge_y, color, scale=0.46, thickness=1)

    return image
