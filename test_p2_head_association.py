import cv2
import numpy as np
import os

from head_association import (
    extract_head_box,
    compute_spatial_overlap,
    associate_riders_with_helmets,
    annotate_individual_riders
)


def create_synthetic_rider_scene(scene_name, rider_configs, bike_box=[100, 200, 700, 550], width=800, height=600):
    """
    Renders synthetic two-wheeler traffic test scenes with precise ground truth
    riders and helmet / no-helmet / unknown configurations.
    
    Args:
        scene_name (str): Label for the test scenario.
        rider_configs (list of dict):
            Each dict: {
                'box': [rx1, ry1, rx2, ry2],
                'conf': 0.90,
                'has_helmet_detection': True/False,
                'is_violation': True/False,
                'detection_conf': 0.88
            }
        bike_box (list): [bx1, by1, bx2, by2]
    """
    img = np.full((height, width, 3), (230, 230, 230), dtype=np.uint8)

    # Road background
    cv2.rectangle(img, (0, int(height * 0.70)), (width, height), (70, 70, 70), -1)
    cv2.line(img, (0, int(height * 0.85)), (width, int(height * 0.85)), (255, 255, 255), 4)

    # Motorcycle silhouette
    bx1, by1, bx2, by2 = bike_box
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (180, 50, 50), -1)
    cv2.circle(img, (bx1 + 80, by2 - 10), 55, (30, 30, 30), -1)
    cv2.circle(img, (bx2 - 80, by2 - 10), 55, (30, 30, 30), -1)
    cv2.putText(img, "MOTORCYCLE", (bx1 + 15, by1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    raw_riders = []
    helmet_detections = []

    for r_cfg in rider_configs:
        rx1, ry1, rx2, ry2 = r_cfg["box"]
        r_conf = r_cfg.get("conf", 0.90)
        rw = rx2 - rx1
        rh = ry2 - ry1

        # Render rider body silhouette
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (120, 120, 120), -1)
        
        # Render head circle in upper 30%
        head_cx = int((rx1 + rx2) / 2)
        head_cy = int(ry1 + 0.15 * rh)
        head_r = int(min(rw, rh * 0.30) / 2) - 4
        
        raw_riders.append({
            "box": [rx1, ry1, rx2, ry2],
            "conf": r_conf
        })

        if r_cfg.get("has_helmet_detection", False):
            is_viol = r_cfg.get("is_violation", False)
            h_conf = r_cfg.get("detection_conf", 0.88)
            label = "without_helmet" if is_viol else "helmet"
            
            # Head bounding box with slight realistic variance
            hx1 = max(0, head_cx - head_r - 4)
            hy1 = max(0, head_cy - head_r - 4)
            hx2 = min(width, head_cx + head_r + 4)
            hy2 = min(height, head_cy + head_r + 4)

            # Draw visual head marker
            color = (0, 0, 200) if is_viol else (0, 180, 0)
            cv2.circle(img, (head_cx, head_cy), head_r, color, -1)

            helmet_detections.append({
                "box": [hx1, hy1, hx2, hy2],
                "conf": h_conf,
                "label": label,
                "is_violation": is_viol
            })
        else:
            # Rider has no helmet detection (occluded or detector missed)
            cv2.circle(img, (head_cx, head_cy), head_r, (160, 160, 160), -1)

    return img, raw_riders, helmet_detections, bike_box


def run_p2_benchmarks():
    print("=" * 65)
    print("   P2: RIDER-TO-HEAD SPATIAL ASSOCIATION BENCHMARK SUITE")
    print("=" * 65)

    os.makedirs("test_samples", exist_ok=True)

    scenarios = [
        # Scenario 1: One Rider with Helmet
        {
            "name": "Scenario 1: One Rider with Helmet",
            "filename": "test_samples/p2_single_rider_helmet.jpg",
            "bike_box": [200, 250, 600, 520],
            "riders": [
                {"box": [340, 120, 480, 460], "has_helmet_detection": True, "is_violation": False, "detection_conf": 0.92}
            ],
            "expected": [("Rider 1", "HELMET_OK")]
        },

        # Scenario 2: Two Riders Mixed (Driver Helmet OK, Pillion No Helmet)
        {
            "name": "Scenario 2: Two Riders Mixed (Helmet OK + No Helmet)",
            "filename": "test_samples/p2_two_riders_mixed.jpg",
            "bike_box": [150, 240, 650, 520],
            "riders": [
                {"box": [240, 110, 370, 450], "has_helmet_detection": True, "is_violation": False, "detection_conf": 0.89},
                {"box": [420, 115, 550, 455], "has_helmet_detection": True, "is_violation": True, "detection_conf": 0.86}
            ],
            "expected": [("Rider 1", "HELMET_OK"), ("Rider 2", "NO_HELMET")]
        },

        # Scenario 3: Three Riders Mixed (Helmet OK + No Helmet + Detector Missed / Unknown)
        {
            "name": "Scenario 3: Three Riders (Helmet OK + No Helmet + Unknown)",
            "filename": "test_samples/p2_three_riders_mixed_unknown.jpg",
            "bike_box": [100, 240, 720, 520],
            "riders": [
                {"box": [180, 100, 310, 450], "has_helmet_detection": True, "is_violation": False, "detection_conf": 0.91},
                {"box": [340, 105, 470, 455], "has_helmet_detection": True, "is_violation": True, "detection_conf": 0.88},
                {"box": [500, 110, 630, 460], "has_helmet_detection": False, "is_violation": False, "detection_conf": 0.0}
            ],
            "expected": [("Rider 1", "HELMET_OK"), ("Rider 2", "NO_HELMET"), ("Rider 3", "UNKNOWN")]
        },

        # Scenario 4: Overlapping Closely-Seated Riders
        {
            "name": "Scenario 4: Overlapping Riders (Spatial Disambiguation)",
            "filename": "test_samples/p2_overlapping_riders.jpg",
            "bike_box": [150, 240, 650, 520],
            "riders": [
                {"box": [270, 110, 390, 450], "has_helmet_detection": True, "is_violation": False, "detection_conf": 0.90},
                {"box": [350, 115, 470, 455], "has_helmet_detection": True, "is_violation": True, "detection_conf": 0.85}
            ],
            "expected": [("Rider 1", "HELMET_OK"), ("Rider 2", "NO_HELMET")]
        }
    ]

    total_scenarios_passed = 0

    for sc in scenarios:
        print("\n" + "-" * 60)
        print(f"RUNNING: {sc['name']}")
        print("-" * 60)

        img, riders, helmets, bike_box = create_synthetic_rider_scene(
            sc["name"],
            sc["riders"],
            bike_box=sc["bike_box"]
        )

        # Execute Rider-to-Head Spatial Association
        associated_riders = associate_riders_with_helmets(
            riders,
            helmets,
            head_ratio=0.30
        )

        motorcycle_entry = [{
            "box": bike_box,
            "conf": 0.95,
            "riders": associated_riders
        }]

        # Annotate image
        annotated = annotate_individual_riders(img.copy(), motorcycle_entry, show_head_zone=True)
        cv2.imwrite(sc["filename"], annotated)
        print(f"Annotated Output Saved: {sc['filename']}")

        print("\nMotorcycle 1:")
        scenario_success = True
        for idx, (expected_tag, expected_status) in enumerate(sc["expected"]):
            actual_r = associated_riders[idx]
            actual_status = actual_r["helmet_status"]
            actual_label = actual_r["status_label"]
            conf_val = actual_r["helmet_conf"]

            is_match = (actual_status == expected_status)
            if not is_match:
                scenario_success = False

            mark = "[PASS]" if is_match else "[FAIL]"
            conf_str = f" (conf: {conf_val:.2f})" if conf_val > 0 else ""
            print(f"  {mark} - Rider {actual_r['rider_id']}: {actual_label}{conf_str} [Expected: {expected_status}]")

        if scenario_success:
            total_scenarios_passed += 1
            print(f"STATUS: PASS")
        else:
            print(f"STATUS: FAIL")

    print("\n" + "=" * 65)
    print(f"BENCHMARK RESULTS: {total_scenarios_passed}/{len(scenarios)} Scenarios Passed")
    print("=" * 65)
    assert total_scenarios_passed == len(scenarios), "Not all P2 benchmark scenarios passed!"


def test_real_project_image():
    print("\n" + "=" * 65)
    print("TEST SUITE: REAL PROJECT IMAGE EVALUATION (test.jpg.jpg)")
    print("=" * 65)

    from ultralytics import YOLO

    vehicle_model = YOLO("yolov8n.pt")
    helmet_model = YOLO("models/helmet_model.pt")

    img_path = "test.jpg.jpg"
    img = cv2.imread(img_path)

    v_results = vehicle_model(img_path, conf=0.25, verbose=False)
    raw_persons = []
    motorcycles = []

    for r in v_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = vehicle_model.names[cls]
            coords = list(map(int, box.xyxy[0]))
            if label == "person":
                raw_persons.append({"box": coords, "conf": conf})
            elif label == "motorcycle":
                motorcycles.append({"box": coords, "conf": conf, "riders": []})

    # Spatial rider association
    for p in raw_persons:
        px1, py1, px2, py2 = p["box"]
        pcx = (px1 + px2) / 2.0
        for b in motorcycles:
            bx1, by1, bx2, by2 = b["box"]
            bw = bx2 - bx1
            if (bx1 - 0.10 * bw) <= pcx <= (bx2 + 0.10 * bw) and py2 >= by1 and py1 <= by2:
                b["riders"].append(p)
                break

    h_results = helmet_model(img_path, conf=0.30, verbose=False)
    helmet_dets = []
    for r in h_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = helmet_model.names[cls].lower().strip()
            is_viol = ("without" in label or "no" in label)
            helmet_dets.append({
                "box": list(map(int, box.xyxy[0])),
                "conf": conf,
                "label": label,
                "is_violation": is_viol
            })

    for b in motorcycles:
        b["riders"] = associate_riders_with_helmets(b["riders"], helmet_dets, head_ratio=0.30)

    print("\nReal Image Detection Output:")
    for idx, b in enumerate(motorcycles, start=1):
        print(f"Motorcycle {idx}:")
        for r in b["riders"]:
            print(f"- Rider {r['rider_id']}: {r['status_label']} (conf: {r['helmet_conf']:.2f})")

    annotated = annotate_individual_riders(img.copy(), motorcycles, show_head_zone=True)
    out_real = "test_samples/real_image_p2_output.jpg"
    cv2.imwrite(out_real, annotated)
    print(f"Saved real annotated output to: {out_real}")


if __name__ == "__main__":
    run_p2_benchmarks()
    test_real_project_image()
