import cv2
import numpy as np


def extract_plate_roi(image, motorcycle_box=None, padding_ratio=0.10):
    """
    Extracts the motorcycle license plate Region of Interest (ROI).
    
    Args:
        image (np.ndarray): Full input scene BGR image.
        motorcycle_box (list or tuple): Bounding box [x1, y1, x2, y2] of the motorcycle.
        padding_ratio (float): Ratio of padding to add around detected plate bounding box.
        
    Returns:
        tuple: (crop_img, (px1, py1, px2, py2))
            crop_img is the cropped plate region.
            (px1, py1, px2, py2) are absolute coordinates in the original image.
            If detection fails, returns (None, None).
    """
    if image is None:
        return None, None

    img_h, img_w = image.shape[:2]

    # 1. Determine vehicle search region
    if motorcycle_box is not None:
        bx1, by1, bx2, by2 = map(int, motorcycle_box)
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(img_w, bx2), min(img_h, by2)
        bw = bx2 - bx1
        bh = by2 - by1

        if bw <= 0 or bh <= 0:
            return None, None

        # Number plates on two-wheelers are located in the lower central/rear sector:
        # Vertical: bottom 45% of motorcycle box
        # Horizontal: central 70% of motorcycle box
        search_y1 = max(0, int(by1 + 0.52 * bh))
        search_y2 = min(img_h, int(by2 + 0.05 * bh))
        search_x1 = max(0, int(bx1 + 0.12 * bw))
        search_x2 = min(img_w, int(bx2 - 0.12 * bw))
    else:
        # Full image search
        search_x1, search_y1, search_x2, search_y2 = 0, 0, img_w, img_h

    search_w = search_x2 - search_x1
    search_h = search_y2 - search_y1

    if search_w < 20 or search_h < 15:
        return None, None

    search_roi = image[search_y1:search_y2, search_x1:search_x2]

    # 2. Edge & Contour Detection for Rectangular License Plate
    gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
    
    # Smooth to reduce noise while preserving edges
    blurred = cv2.bilateralFilter(gray_roi, 9, 75, 75)
    
    # Gradient in horizontal direction (detect vertical character edges)
    sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel_x)
    
    # Morphological closing to group characters into a single plate region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(sobel_abs, cv2.MORPH_CLOSE, kernel)
    
    # Binarize
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_candidate = None
    max_score = -1.0
    
    roi_area = float(search_w * search_h)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        
        # Indian two-wheeler plates typically have aspect ratio between 1.5 and 4.8
        # and represent a reasonable fraction of the lower motorcycle region
        if 1.5 <= aspect_ratio <= 5.0 and (0.01 * roi_area <= area <= 0.45 * roi_area):
            # Prefer rectangular density
            contour_area = cv2.contourArea(cnt)
            solidity = contour_area / float(area) if area > 0 else 0
            
            # Distance from center horizontal axis
            center_dist = abs((x + w / 2.0) - (search_w / 2.0)) / float(search_w)
            
            score = (area / roi_area) * 2.0 + solidity - center_dist
            
            if score > max_score:
                max_score = score
                best_candidate = (x, y, w, h)
                
    # 3. Fallback: If no sharp rectangular contour was identified, use central lower sub-box
    if best_candidate is not None:
        cx, cy, cw, ch = best_candidate
        plate_x1 = search_x1 + cx
        plate_y1 = search_y1 + cy
        plate_x2 = plate_x1 + cw
        plate_y2 = plate_y1 + ch
    elif motorcycle_box is not None:
        # Standard default plate location on rear mudguard
        plate_x1 = int(search_x1 + 0.20 * search_w)
        plate_x2 = int(search_x2 - 0.20 * search_w)
        plate_y1 = int(search_y1 + 0.35 * search_h)
        plate_y2 = int(search_y1 + 0.85 * search_h)
    else:
        return None, None

    # 4. Add safety padding around plate bounding box
    pad_w = int((plate_x2 - plate_x1) * padding_ratio)
    pad_h = int((plate_y2 - plate_y1) * padding_ratio)
    
    px1 = max(0, plate_x1 - pad_w)
    py1 = max(0, plate_y1 - pad_h)
    px2 = min(img_w, plate_x2 + pad_w)
    py2 = min(img_h, plate_y2 + pad_h)
    
    crop = image[py1:py2, px1:px2]
    
    if crop is None or crop.shape[0] < 10 or crop.shape[1] < 15:
        return None, None
        
    return crop, (px1, py1, px2, py2)


def preprocess_plate_image(crop_bgr, target_height=100):
    """
    Applies multi-stage image enhancement to boost OCR accuracy on
    blurry, low-resolution, or low-light license plate crops.
    
    Returns:
        dict: Processed image variants:
            - 'enhanced': CLAHE contrast-enhanced grayscale.
            - 'sharpened': High-pass edge sharpened grayscale.
            - 'otsu': Binarized Otsu threshold.
            - 'adaptive': Adaptive Gaussian threshold.
            - 'resized_bgr': High-resolution scaled BGR.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {}

    h, w = crop_bgr.shape[:2]
    
    # 1. High-Quality Upscaling
    if h < target_height:
        scale = float(target_height) / float(h)
        new_w = max(int(w * scale), 160)
        resized_bgr = cv2.resize(
            crop_bgr,
            (new_w, target_height),
            interpolation=cv2.INTER_CUBIC
        )
    else:
        resized_bgr = crop_bgr.copy()

    # 2. Grayscale
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Bilateral Filter Denoising (removes sensor grain while keeping character edges crisp)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Compensates for shadows, headlights, and low-light conditions
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Unsharp Masking / Sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # 6. Thresholding variants
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        4
    )

    return {
        "enhanced": enhanced,
        "sharpened": sharpened,
        "otsu": otsu,
        "adaptive": adaptive,
        "resized_bgr": resized_bgr
    }


def detect_license_plate_roi(image, motorcycle_box=None):
    """
    Wrapper for plate ROI localization returning a dictionary.
    """
    crop, box = extract_plate_roi(image, motorcycle_box=motorcycle_box)
    if box is not None:
        return {"found": True, "box": box, "crop": crop}
    return {"found": False, "box": None, "crop": None}
