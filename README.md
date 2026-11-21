# Two-Wheeler Traffic Violation AI Platform (TwoWheelerViolationAI)

An end-to-end, AI-powered traffic surveillance and automated e-Challan enforcement platform for two-wheelers. Built with **YOLOv8**, custom helmet detection, **ByteTrack** multi-object tracking, position-aware **EasyOCR** with Indian registration syntax regex correction, **Google Gemini 2.5 Flash** multimodal vision fallback, an **SQLite** database, an **Advanced ReportLab PDF e-Challan Generator (P6)** with demonstration QR verification, and a modern **Streamlit Web Dashboard (P5)**.

---

## 🌟 Key Features & Architecture (P1 to P6)

### 1. License Plate ROI Extraction & Position-Aware Indian Plate Regex (P1)
- **Module:** `plate_roi.py` & `plate_ocr.py`
- Automatic motorcycle number plate ROI localization, aspect ratio filtering, and 10% margin padding.
- OpenCV preprocessing: Bicubic super-resolution, bilateral denoising, CLAHE contrast enhancement, and Otsu thresholding.
- Indian vehicle registration format validation (`^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$`).
- **Position-aware character correction**:
  - State & Series positions (Letters): `0 -> O`, `8 -> B`, `1 -> I`, `5 -> S`, `2 -> Z`
  - District & Number positions (Digits): `O -> 0`, `B -> 8`, `I -> 1`, `S -> 5`, `Z -> 2`
- High Security Registration Plate (`IND`) emblem stripping and state code verification.

### 2. Rider-to-Head Spatial Association (P2)
- **Module:** `head_association.py` & `helmet_detection.py`
- Identifies the upper 30% of each rider bounding box as the anatomical head search region.
- Associates helmet detections (`HELMET` vs `NO_HELMET`) using greedy bipartite overlap matching.
- Assigns unique rider IDs (`Rider 1`, `Rider 2`, `Rider 3`) and individual color-coded badges (🟢 Green: OK, 🔴 Red: Violation, 🟡 Amber: Unknown).

### 3. Gemini Multimodal Fallback for Uncertain Plates (P3)
- **Module:** `multimodal_verifier.py`
- Official Google GenAI SDK integration (`gemini-2.5-flash` / `gemini-1.5-flash`).
- Automatically triggered when EasyOCR confidence is `< 0.50` or format is ambiguous.
- Strict structured JSON response parsing (`detected_number`, `confidence`, `is_readable`, `explanation`).
- Conservative reconciliation logic: flags mismatches for officer review (`REVIEW_REQUIRED`) rather than hallucinating digits.

### 4. Video Processing & ByteTrack Multi-Object Tracking (P4)
- **Module:** `video_processor.py`
- Supports video files (MP4, AVI) and live RTSP CCTV camera streams.
- Persistent vehicle tracking across frames using **ByteTrack** (`bytetrack.yaml`).
- 60-second cooldown deduplication per vehicle tracking ID and license plate to prevent spamming duplicate fines.
- Automatic violation evidence frame snapshotting to `evidence/`.

### 5. Interactive Streamlit Web Dashboard & SQLite Database (P5)
- **Module:** `app.py`, `database.py`, `dashboard_pages/`, `ui_styles.py`
- Professional dark theme with custom responsive CSS.
- **5 Integrated Pages**:
  1. **Dashboard Overview**: Real-time KPI metrics, AI engine status, recent live violations feed.
  2. **Image Detection**: Upload images or benchmark samples, run the full pipeline, inspect individual rider cards, view plate crop, and issue e-Challan.
  3. **Video Detection**: Live surveillance feed with interactive start/stop, real-time ByteTrack tracking boxes, live violation counters, and evidence snapshot gallery.
  4. **Challan Management**: Searchable and filterable registry of all stored SQLite records, CSV export, photographic evidence viewer, and ReportLab PDF download.
  5. **Analytics & Reports**: Offense category distributions, repeat offender rankings, source breakdown, and revenue statistics.

### 6. Advanced PDF Challan Generator (P6)
- **Module:** `pdf_challan_generator.py` & `fine_generator.py`
- **Official Statutory Legal Provisions**: Strictly verified against the Gazette of India for the Motor Vehicles Act 1988 (as amended by Act No. 32 of 2019):
  - **Section 129 read with Sec 194D**: Riding two-wheeler without protective headgear (helmet) — ₹1,000 fine per rider + 3-month license disqualification.
  - **Section 128 read with Sec 194C**: More than one pillion rider (triple riding) — ₹1,000 fine + 3-month license disqualification.
  - **Section 194A**: Carriage of excess passengers (Dangerous overloading >3 persons) — ₹2,000 fine.
  - **Section 136A**: Electronic monitoring and surveillance for road safety.
- **Dual Photographic Evidence Panel**: Embeds both the full scene surveillance photograph and the extracted license plate ROI crop side-by-side with spatial rider-head status breakdown.
- **Demonstration QR Code**: Includes a scannable demonstration QR code linked to a simulated verification record with a cryptographic SHA-256 audit token. Explicitly non-payable and marked `DEMO ONLY • NON-PAYABLE`.
- **Academic & Legal Disclaimers**: Prominent system-generated disclaimer banners clarifying that this is an academic research project, carries no legal force, and does not collect real money.
- **Organized Storage & Export**: Automatically saves generated PDF reports in `challans/pdf_reports/` and provides instant in-memory downloads from the Streamlit web dashboard.
- **Robust Fault Tolerance**: Gracefully handles missing images, corrupted files, and incomplete records using PIL-rendered placeholder graphics.

---

## 💻 Running Locally in VS Code

### 1. Open Project in VS Code
1. Extract `TwoWheelerViolationAI_Final.zip`.
2. Open VS Code, select **File > Open Folder...**, and choose the extracted `TwoWheelerViolationAI` directory.

### 2. Set Up Python Virtual Environment
Open the integrated terminal in VS Code (`Ctrl + ~`) and run:
```bash
# Windows (PowerShell / Command Prompt)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
copy .env.example .env    # Windows
cp .env.example .env      # Linux/macOS
```
*(Optional) Add your Google AI Studio API key (`GEMINI_API_KEY=your_key`) if you wish to use the live cloud Gemini multimodal vision fallback. If left blank, the system automatically uses offline local EasyOCR.*

---

## 🚀 Execution Commands

### 1. Launch the Streamlit Web Dashboard (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Image Surveillance via CLI
```bash
python main.py --image test_with_plate.jpg
```

### 3. Run Video Multi-Object Tracking via CLI
```bash
python main.py --video test_traffic.mp4 --output tracked_traffic_output.mp4
```

### 4. Run Headless / RTSP CCTV Stream
```bash
python main.py --video rtsp://admin:pass@192.168.1.100:554/live --no-gui
```

### 5. Generate Standalone Demonstration PDF Reports
```bash
python pdf_challan_generator.py
```

---

## 🧪 Testing Instructions

Run the complete automated test suite with a single command:
```bash
python test_full_integration_pipeline.py
```

Run individual feature test suites:
```bash
python test_p1_plate_recognition.py       # P1: ROI extraction & regex correction
python test_p2_head_association.py        # P2: Spatial head mapping & helmet audit
python test_p3_gemini_fallback.py         # P3: Gemini vision fallback & reconciliation
python test_p4_video_tracking.py          # P4: ByteTrack video tracking & deduplication
python test_p5_dashboard.py               # P5: Database persistence & telemetry
python test_p6_pdf_generator.py           # P6: PDF report formatting & QR scanning
```

---

## 📂 Project Directory Structure

```text
TwoWheelerViolationAI/
├── app.py                         # Streamlit Web Dashboard entry point
├── ui_styles.py                   # Modern dark theme styles & CSS components
├── database.py                    # SQLite database layer (traffic_violations.db)
├── pdf_challan_generator.py       # Advanced ReportLab PDF e-Challan generator (P6)
├── dashboard_pages/               # Modular Streamlit page views
│   ├── overview.py                # 1. Executive Dashboard & KPIs
│   ├── image_detection.py         # 2. Image Detection & Spatial Analysis
│   ├── video_detection.py         # 3. Video Tracking & ByteTrack Stream
│   ├── challan_management.py      # 4. Challan Search, Filter & PDF Download
│   └── analytics.py               # 5. Statistical Analytics & Repeat Offenders
├── main.py                        # Main CLI orchestrator
├── video_processor.py             # ByteTrack video tracker & deduplicator (P4)
├── head_association.py            # Spatial Rider-to-Head 30% mapper (P2)
├── helmet_detection.py            # Helmet detection & individual rider scoring
├── plate_roi.py                   # License plate ROI detection & preprocessing (P1)
├── plate_ocr.py                   # Position-aware regex correction & OCR (P1)
├── multimodal_verifier.py         # Google Gemini Vision API fallback (P3)
├── fine_generator.py              # MV Act fine calculation & text receipt
├── detect.py                      # Base YOLO detection utilities
├── triple_riding.py               # Overloading detection utilities
├── models/
│   └── helmet_model.pt            # Custom YOLOv8 helmet classifier (6.2 MB)
├── yolov8n.pt                     # Base YOLOv8 object detector (6.55 MB)
├── evidence/                      # Auto-saved violation evidence frames
├── challans/                      # Saved receipts & organized reports
│   └── pdf_reports/               # Generated PDF e-Challan notices
├── traffic_violations.db          # Persistent SQLite database
├── test_traffic.mp4               # Benchmark test traffic video
├── test_with_plate.jpg            # Benchmark test traffic image
├── test_samples/                  # Isolated benchmark scenario images
├── requirements.txt               # Pinned Python dependencies
├── .env.example                   # Environment configuration template
├── ARCHITECTURE.md                # System design & architecture specification
└── README.md                      # Complete user & developer documentation
```

---

## ⚖️ Statutory Motor Vehicles Act Penalties (Verified Official Citations)

| Offense | Section (M.V. Act 1988/2019) | Statutory Penalty | Legal Consequence |
|---|---|---|---|
| **Riding Without Helmet** | Section 129 read with Section 194D | ₹ 1,000 per unhelmeted rider | 3-month driving license disqualification |
| **Triple Riding (3 Riders)** | Section 128 read with Section 194C | ₹ 1,000 compound fine | 3-month driving license disqualification |
| **Dangerous Overloading (>3 Riders)** | Section 194A | ₹ 2,000 compound fine | Overloading liability |
| **Electronic Surveillance Notice** | Section 136A | Statutory framework | Electronic monitoring & evidence recording |
