import cv2
import numpy as np
import os


def generate_traffic_video(output_path="test_traffic.mp4", num_frames=60, width=800, height=500, fps=25):
    """
    Synthesizes a realistic traffic camera video sequence featuring two-wheelers
    moving along a multi-lane roadway to benchmark ByteTrack multi-object tracking.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    base_bike_img = cv2.imread("test_with_plate.jpg")
    if base_bike_img is None:
        base_bike_img = cv2.imread("test.jpg.jpg")

    # Resize motorcycle crop for natural traffic scaling
    target_bw = 260
    target_bh = int(base_bike_img.shape[0] * (target_bw / base_bike_img.shape[1]))
    bike_sprite = cv2.resize(base_bike_img, (target_bw, target_bh))

    # Second motorcycle sprite (clean rider)
    clean_bike_img = cv2.imread("test_samples/p2_single_rider_helmet.jpg")
    clean_bw = 200
    clean_bh = int(clean_bike_img.shape[0] * (clean_bw / clean_bike_img.shape[1]))
    clean_sprite = cv2.resize(clean_bike_img, (clean_bw, clean_bh))

    for frame_i in range(num_frames):
        # 1. Asphalt Road background
        frame = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)
        
        # Road lane dividers (dashed yellow and white lines)
        cv2.line(frame, (0, 120), (width, 120), (220, 220, 220), 3)
        cv2.line(frame, (0, 380), (width, 380), (220, 220, 220), 3)

        dash_offset = (frame_i * 8) % 60
        for x in range(-60, width + 60, 60):
            cv2.line(frame, (x + dash_offset, 250), (x + dash_offset + 30, 250), (0, 220, 255), 3)

        # 2. Vehicle 1: Violating Motorcycle (moves left to right in lower lane)
        # Position progresses smoothly
        vx1 = int(20 + frame_i * 7.5)
        vy1 = int(180 + np.sin(frame_i * 0.1) * 6)

        # Alpha composite or paste sprite
        if 0 <= vx1 < width - target_bw and 0 <= vy1 < height - target_bh:
            frame[vy1:vy1 + target_bh, vx1:vx1 + target_bw] = bike_sprite

        # 3. Vehicle 2: Compliant Motorcycle (moves slower in upper lane)
        vx2 = int(100 + frame_i * 4.5)
        vy2 = int(60 + np.cos(frame_i * 0.08) * 4)
        if 0 <= vx2 < width - clean_bw and 0 <= vy2 < height - clean_bh:
            frame[vy2:vy2 + clean_bh, vx2:vx2 + clean_bw] = clean_sprite

        # Add timestamp watermarking (typical CCTV feed)
        cv2.putText(
            frame,
            f"CCTV-CAM-04 | 21-11-2026 13:30:{frame_i // 25:02d}.{frame_i % 25:02d}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        out.write(frame)

    out.release()
    print(f"Generated synthetic traffic video -> {output_path} ({num_frames} frames, {width}x{height} @ {fps}fps)")


if __name__ == "__main__":
    generate_traffic_video("test_traffic.mp4", num_frames=50)
