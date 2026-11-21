import cv2
import os
import subprocess
import time

from video_processor import VideoTrafficAuditor, process_traffic_video


def run_video_tracking_test():
    print("=" * 65)
    print("TEST SUITE 1: BYTETRACK MULTI-OBJECT TRACKING & DEDUPLICATION")
    print("=" * 65)

    video_file = "test_traffic.mp4"
    assert os.path.exists(video_file), f"Test video not found: {video_file}"

    # Clear previous evidence in test folder to measure new files
    evidence_dir = "evidence"
    os.makedirs(evidence_dir, exist_ok=True)
    initial_evidence_count = len([f for f in os.listdir(evidence_dir) if f.endswith(".jpg")])

    auditor = VideoTrafficAuditor(
        cooldown_seconds=60,
        ocr_interval_frames=5,
        evidence_dir=evidence_dir
    )

    summary = auditor.process_video(
        video_source=video_file,
        output_video_path="tracked_traffic_output.mp4",
        show_live=False,  # Headless test run
        max_frames=50,
        enable_gemini_fallback=False  # Local test
    )

    new_evidence_count = len([f for f in os.listdir(evidence_dir) if f.endswith(".jpg")])

    print("\nAuditor Test Results Verification:")
    print(f"  • Frames Processed        : {summary['total_frames_processed']} (Expected: 50)")
    print(f"  • Unique Vehicles Tracked : {summary['unique_vehicles_tracked']} (Expected: >= 1)")
    print(f"  • Violations Logged       : {summary['unique_violations_recorded']}")
    print(f"  • Confirmed Challans      : {summary['confirmed_challans_issued']}")
    print(f"  • Evidence Files Created  : {new_evidence_count - initial_evidence_count}")

    # Deduplication assertion: We processed 50 frames with a violating vehicle.
    # The deduplication system must ensure that the number of unique violations logged
    # matches unique vehicles (1 or 2) and NOT 50!
    assert summary["total_frames_processed"] == 50, "Frame count mismatch!"
    assert summary["unique_vehicles_tracked"] >= 1, "No vehicles were tracked by ByteTrack!"
    assert summary["unique_violations_recorded"] <= 2, f"Deduplication failed! Generated {summary['unique_violations_recorded']} violations instead of 1."
    print("\n[PASS] ByteTrack tracking and Event-Deduplication validated successfully!")


def run_corrupted_and_stream_handling_test():
    print("\n" + "=" * 65)
    print("TEST SUITE 2: ERROR HANDLING & DISCONNECTED STREAM ROBUSTNESS")
    print("=" * 65)

    # 1. Non-existent file / stream
    auditor = VideoTrafficAuditor()
    summary_missing = auditor.process_video("non_existent_stream.mp4", show_live=False)
    assert summary_missing["total_frames_processed"] == 0, "Failed to handle missing video gracefully!"
    print("  [PASS] Non-existent video handled gracefully without crashing.")

    # 2. Corrupted file test
    corrupted_path = "corrupted_test.mp4"
    with open(corrupted_path, "wb") as f:
        f.write(b"NOT_A_VALID_VIDEO_STREAM_BYTES_CORRUPTED_12345")

    summary_corrupt = auditor.process_video(corrupted_path, show_live=False)
    assert summary_corrupt["total_frames_processed"] == 0, "Failed to handle corrupted video gracefully!"
    print("  [PASS] Corrupted video file handled gracefully without crashing.")
    if os.path.exists(corrupted_path):
        os.remove(corrupted_path)


def run_main_integration_test():
    print("\n" + "=" * 65)
    print("TEST SUITE 3: MAIN.PY VIDEO CLI INTEGRATION")
    print("=" * 65)

    cmd = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
        "main.py",
        "--video", "test_traffic.mp4",
        "--no-gui"
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, f"main.py --video failed: {p.stderr}"
    print("  [PASS] main.py --video executed successfully with returncode 0.")
    print("  Snippet of main.py video summary output:")
    for line in p.stdout.strip().split("\n")[-12:]:
        print(f"    {line}")


if __name__ == "__main__":
    run_video_tracking_test()
    run_corrupted_and_stream_handling_test()
    run_main_integration_test()
    print("\n" + "=" * 65)
    print("ALL P4 VIDEO & TRACKING TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
