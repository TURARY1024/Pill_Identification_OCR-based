#Rushi

from collections import Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import colorsys

def rotate_image_by_angle(image, angle):
    """
    將圖片依指定角度旋轉。
    - image: 輸入圖片（OpenCV 格式）
    - angle: 順時針旋轉角度（例如 90, 180）
    - return: 旋轉後的圖片
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def enhance_contrast(img, clip_limit, alpha, beta):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhance_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhance_img, (5, 5), 3.0)
    return cv2.addWeighted(enhance_img, alpha, blurred, beta, 0)

def get_center_region(img, size=100):
    """
    擷取圖片的中央區域 (固定大小)。
    - img: 輸入圖片 (H, W, C)
    - size: 方形區域邊長 (像素)，預設 100
    - return: 中央裁切後的圖片
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2 # 圖片中心點
    
    # 計算邊界
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)
    
    return img[y1:y2, x1:x2]

def increase_brightness(img, value=30):
    """Increase brightness of an RGB image by boosting V channel in HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img_bright

# ===== NEW HSV COLOR RECOGNITION SYSTEM =====

def extract_pill_colors_hsv(cropped_img, contour=None, visualize=True):
    """
    Robust HSV color extraction:
    - contour mask if available
    - remove low-V (background) and very dark imprint pixels
    - erode edges to avoid background bleed
    - use robust (median-based) HSV stats
    """
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # Base mask
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    if contour is not None and len(contour) >= 5:
        cv2.fillPoly(mask, [contour], 255)
        analysis_method = "Contour-based"
    else:
        mask[:] = 255
        analysis_method = "Brightness-based"
    
    h, s, v = cv2.split(hsv_img)
    
    # Compute local thresholds only within current mask
    v_in = v[mask > 0]
    if v_in.size == 0:
        return [], None, analysis_method
    
    # Remove background: drop darkest 15% within mask
    v_thresh = np.percentile(v_in, 15)
    mask = cv2.bitwise_and(mask, (v > v_thresh).astype(np.uint8) * 255)
    
    # Remove very dark imprint pixels
    mask = cv2.bitwise_and(mask, (v > int(0.25 * 255)).astype(np.uint8) * 255)
    
    # Slight erosion to avoid edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    hsv_pixels = hsv_img[mask > 0]
    if hsv_pixels.size == 0:
        return [], None, analysis_method
    
    # --- Additional mask cleanups ---
    # 1) Remove specular highlights (very bright pixels)
    mask = cv2.bitwise_and(mask, (v < int(0.98 * 255)).astype(np.uint8) * 255)
    
    # 2) Extra erosion when mask too large (reduce edge bleed)
    mask_area_ratio = float(np.count_nonzero(mask)) / mask.size
    if mask_area_ratio > 0.60:
        mask = cv2.erode(mask, kernel, iterations=1) # one more erosion
    
    # 3) Exclude a thin border band (5% per side) to prevent background intrusion
    h_img, w_img = mask.shape
    bx = int(0.05 * w_img)
    by = int(0.05 * h_img)
    border_mask = np.zeros_like(mask)
    border_mask[by:h_img-by, bx:w_img-bx] = 255
    mask = cv2.bitwise_and(mask, border_mask)
    
    # 4) Imprint exclusion: remove very dark imprint pixels + slight dilation
    imprint = (v < int(0.25 * 255)).astype(np.uint8) * 255
    imprint = cv2.dilate(imprint, kernel, iterations=1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(imprint))
    
    # If mask empty, bail as you do now
    hsv_pixels = hsv_img[mask > 0]
    if hsv_pixels.size == 0:
        return [], None, analysis_method
    
    # Pixel-count guard
    min_pixels = 2000
    if hsv_pixels.shape[0] < min_pixels:
        # build center-only mask and retry
        h_i, w_i = mask.shape
        cx0, cy0 = int(0.3 * w_i), int(0.3 * h_i)
        cx1, cy1 = int(0.7 * w_i), int(0.7 * h_i)
        center = np.zeros_like(mask)
        center[cy0:cy1, cx0:cx1] = 255
        mask_center = cv2.bitwise_and(mask, center)
        hsv_pixels_center = hsv_img[mask_center > 0]
        if hsv_pixels_center.size >= min_pixels:
            mask = mask_center
            hsv_pixels = hsv_pixels_center
    
    # Robust (median) HSV statistics
    # Extract raw channels
    h_vals = hsv_pixels[:, 0].astype(np.float32)
    s_vals = hsv_pixels[:, 1].astype(np.float32)
    v_vals = hsv_pixels[:, 2].astype(np.float32)
    
    # Adaptive bins for low-sat scenes
    s_med_tmp = float(np.median(s_vals))
    s_norm_tmp = s_med_tmp / 255.0
    bins = 24 if s_norm_tmp < 0.12 else 12
    
    # Hue histogram -> mode bin then median inside
    hist, edges = np.histogram(h_vals, bins=bins, range=(0, 180))
    bin_idx = int(np.argmax(hist))
    h_lo, h_hi = edges[bin_idx], edges[bin_idx + 1]
    h_sel = h_vals[(h_vals >= h_lo) & (h_vals < h_hi)]
    h_med = float(np.median(h_sel)) if h_sel.size > 0 else float(np.median(h_vals))
    
    # IQR-trimmed medians for S and V
    Q1_s, Q3_s = np.percentile(s_vals, [25, 75])
    Q1_v, Q3_v = np.percentile(v_vals, [25, 75])
    s_core = s_vals[(s_vals >= Q1_s) & (s_vals <= Q3_s)]
    v_core = v_vals[(v_vals >= Q1_v) & (v_vals <= Q3_v)]
    s_med = float(np.median(s_core)) if s_core.size > 0 else float(np.median(s_vals))
    v_med = float(np.median(v_core)) if v_core.size > 0 else float(np.median(v_vals))
    
    avg_hsv = np.uint8([[[h_med, s_med, v_med]]])
    
    # Classify into semantic Chinese color
    dominant_color = classify_hsv_to_semantic_color(h_med, s_med, v_med)
    dominant_color = norm_color(dominant_color) # normalized output
    
    return [dominant_color], avg_hsv, analysis_method

def classify_hsv_to_semantic_color(h, s, v):
    """
    HSV -> Chinese color labels.
    Tuned thresholds for real-world lighting.
    """
    h_deg = float(h) * 2.0
    s_norm = float(s) / 255.0
    v_norm = float(v) / 255.0
    
    # Black
    if v_norm < 0.18:
        return "黑色"
    
    # Strong white protection
    if (s_norm < 0.10 and v_norm > 0.88) or (s_norm < 0.14 and v_norm > 0.90):
        return "白色"
    
    # Gray window (narrower; keep above white thresholds)
    if s_norm < 0.18 and 0.58 < v_norm < 0.90:
        return "灰色"
    
    # Brown: only when saturated + darker + warm hues
    if s_norm > 0.25 and v_norm < 0.60 and 10 <= h_deg < 60:
        return "棕色"
    
    # Require minimum saturation for chromatic classes
    if s_norm < 0.18:
        # Low chroma: if bright => white, else gray
        if v_norm >= 0.90:
            return "白色"
        if v_norm > 0.58:
            return "灰色"
        # else darker low-chroma → tends to brown/black already handled
        return "灰色"
    
    # Hue bands (only if s_norm >= 0.18)
    if h_deg < 15 or h_deg >= 345:
        return "紅色"
    elif h_deg < 42:
        # Warm-white bias: prevent false orange/yellow on whites
        if v_norm > 0.90 and s_norm < 0.16:
            return "白色"
        return "橘色"
    elif h_deg < 70:
        if v_norm > 0.90 and s_norm < 0.16:
            return "白色"
        return "黃色"
    elif h_deg < 170:
        return "綠色"
    elif h_deg < 250:
        return "藍色"
    elif h_deg < 290:
        return "紫色"
    elif h_deg < 345:
        return "粉紅色"
    
    # Default
    return "其他"

def norm_color(s: str) -> str:
    """Normalize color names for safer comparison."""
    return str(s).strip().replace(' ', '')

def is_color_similar_semantic(a, b, similar_fn):
    """Symmetric similarity check between two Chinese color labels."""
    if not a or not b:
        return False
    a = norm_color(a)
    b = norm_color(b)
    if a == b:
        return True
    ta = similar_fn(a)["similar_colors"]
    tb = similar_fn(b)["similar_colors"]
    return (a in tb) or (b in ta)

def get_color_tolerance(color):
    """
    Define color tolerance for Chinese color names
    """
    similar_colors = {
        "紅色": ["粉紅色", "橘色"],
        "藍色": ["紫色"],
        "黃色": ["橘色", "綠色"],
        "白色": ["灰色"],
        "黑色": ["棕色", "灰色"],
        "綠色": ["黃色"],
        "紫色": ["藍色", "粉紅色"],
        "粉紅色": ["紅色", "紫色"],
        "橘色": ["紅色", "黃色"],
        "棕色": ["黑色"],
        "灰色": ["白色", "黑色"]
    }
    
    return {
        "similar_colors": similar_colors.get(color, []),
        "tolerance_range": 0.15
    }

# def display_color_analysis_hsv(cropped_img, hsv_img, avg_hsv, dominant_color, method):
#     """
#     Display HSV color analysis results
#     """
#     plt.figure(figsize=(15, 4))
    
#     # Original image
#     plt.subplot(141)
#     plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.title("Original Pill Image")
    
#     # HSV visualization
#     plt.subplot(142)
#     plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
#     plt.axis('off')
#     plt.title("HSV Representation")
    
#     # Dominant color patch
#     plt.subplot(143)
#     color_patch = np.full((100, 100, 3), cv2.cvtColor(avg_hsv, cv2.COLOR_HSV2RGB)[0][0], dtype=np.uint8)
#     plt.imshow(color_patch)
#     plt.axis('off')
#     plt.title(f"Dominant Color\n{dominant_color}")
    
#     # Analysis info
#     plt.subplot(144)
#     plt.text(0.1, 0.8, f"Method: {method}", fontsize=12, transform=plt.gca().transAxes)
#     plt.text(0.1, 0.6, f"Color: {dominant_color}", fontsize=14, weight='bold', transform=plt.gca().transAxes)
#     plt.text(0.1, 0.4, f"HSV: {avg_hsv[0][0]}", fontsize=10, transform=plt.gca().transAxes)
#     plt.axis('off')
#     plt.title("Analysis Results")
    
#     plt.tight_layout()
#     plt.show()

# ===== SHAPE DETECTION (UNCHANGED) =====

# === 可調參數（預設值放你目前最佳）===
CIRCLE_LO = 1
CIRCLE_HI = 1.2
ELLIPSE_HI = 3.8

def set_shape_thresholds(circle_lo: float, circle_hi: float, ellipse_hi: float):
    global CIRCLE_LO, CIRCLE_HI, ELLIPSE_HI
    CIRCLE_LO = circle_lo
    CIRCLE_HI = circle_hi
    ELLIPSE_HI = ellipse_hi

def detect_shape_three_classes(contour, expected_shape=None):
    set_shape_thresholds(CIRCLE_LO, CIRCLE_HI, ELLIPSE_HI)
    shape = "其他"
    try:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major, minor = axes
            if minor == 0:
                return shape
            
            ratio = max(major, minor) / min(major, minor)
            ratios_list.append(ratio)
            
            # === classify with global thresholds ===
            if CIRCLE_LO <= ratio <= CIRCLE_HI:
                shape = "圓形"
            elif ratio <= ELLIPSE_HI:
                shape = "橢圓形"
            else:
                shape = "其他"
                
    except Exception as e:
        print(f"❗ detect_shape_three_classes 發生錯誤：{e}")
        
    return shape

# === 增強處理函式 ===
def desaturate_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s.fill(0)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def enhance_for_blur(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    contrast_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 1.0)
    sharpened = cv2.addWeighted(contrast_img, 1.8, blurred, -0.8, 0)
    return cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

def preprocess_with_shadow_correction(img_bgr):
    """改進的前處理，更好地分離藥物與背景"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 多尺度的背景估計
    blur1 = cv2.GaussianBlur(gray, (25, 25), 0)
    blur2 = cv2.GaussianBlur(gray, (75, 75), 0)
    
    # 雙重陰影校正
    corrected1 = cv2.divide(gray, blur1, scale=255)
    corrected2 = cv2.divide(gray, blur2, scale=255)
    corrected = cv2.addWeighted(corrected1, 0.5, corrected2, 0.5, 0)
    
    # 使用 OTSU 自動找最佳閾值
    _, otsu_thresh = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形態學操作去除雜訊
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def detect_shape_from_image(cropped_img, original_img=None, expected_shape=None):
    try:
        output = cropped_img.copy()
        thresh = preprocess_with_shadow_correction(output)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        shape = "其他"
        
        if not contours and original_img is not None:
            gray_fallback = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, thresh_fallback = cv2.threshold(gray_fallback, 127, 255, cv2.THRESH_BINARY)
            contours_fallback, _ = cv2.findContours(thresh_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_fallback:
                main_contour = max(contours_fallback, key=cv2.contourArea)
                shape = detect_shape_three_classes(main_contour, expected_shape=expected_shape)
        
        elif contours:
            main_contour = max(contours, key=cv2.contourArea)
            shape = detect_shape_three_classes(main_contour, expected_shape=expected_shape)
        
        if expected_shape:
            result = "✅" if shape == expected_shape else "❌"
            return shape, result
        
        return shape, None
        
    except Exception as e:
        print(f"❗ 發生錯誤：{e}")
        return "錯誤", None

# ===== COMBINED DETECTION FUNCTION =====
def detect_shape_and_extract_colors(cropped_img, original_img=None, debug=False):
    """
    Combined shape and color analysis using HSV color recognition
    """
    try:
        output = cropped_img.copy()
        thresh = preprocess_with_shadow_correction(output)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours and original_img is not None:
            gray_fallback = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, thresh_fallback = cv2.threshold(gray_fallback, 127, 255, cv2.THRESH_BINARY)
            contours_fallback, _ = cv2.findContours(thresh_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_fallback:
                main_contour = max(contours_fallback, key=cv2.contourArea)
                shape = detect_shape_three_classes(main_contour)
                colors, hsv_avg, method = extract_pill_colors_hsv(original_img, main_contour, visualize=debug)
                return shape, colors, hsv_avg, method
            else:
                # Center-crop fallback when no contours from fallback
                h_i, w_i = cropped_img.shape[:2]
                x0, y0 = int(w_i * 0.2), int(h_i * 0.2)
                x1, y1 = int(w_i * 0.8), int(h_i * 0.8)
                center_crop = cropped_img[y0:y1, x0:x1]
                fallback_colors, fallback_hsv, fallback_method = extract_pill_colors_hsv(center_crop, contour=None, visualize=debug)
                if fallback_colors:
                    return "其他", fallback_colors, fallback_hsv, f"{fallback_method}_CenterCrop"
                return "其他", [], None, "Failed"
                
        elif contours:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            img_area = cropped_img.shape[0] * cropped_img.shape[1]
            area_ratio = area / img_area
            
            # Validate segmentation quality
            is_valid = validate_color_analysis(contours, area_ratio)
            
            if not is_valid:
                shape = detect_shape_three_classes(main_contour)
                # Fallback: center-focused crop HSV if segmentation invalid
                h_i, w_i = cropped_img.shape[:2]
                x0, y0 = int(w_i * 0.2), int(h_i * 0.2)
                x1, y1 = int(w_i * 0.8), int(h_i * 0.8)
                center_crop = cropped_img[y0:y1, x0:x1]
                fallback_colors, fallback_hsv, fallback_method = extract_pill_colors_hsv(center_crop, contour=None, visualize=debug)
                if fallback_colors:
                    return shape, fallback_colors, fallback_hsv, f"{fallback_method}_CenterCrop"
                return shape, [], None, "Invalid_Segmentation"
            
            shape = detect_shape_three_classes(main_contour)
            colors, hsv_avg, method = extract_pill_colors_hsv(cropped_img, main_contour, visualize=debug)
            
            if debug:
                cv2.drawContours(output, [main_contour], -1, (0, 0, 255), 3)
                x, y, w, h = cv2.boundingRect(main_contour)
                cv2.putText(output, f"{shape} | {colors[0] if colors else 'No Color'}",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return shape, colors, hsv_avg, method
        else:
            # No contours at all: center-crop fallback
            h_i, w_i = cropped_img.shape[:2]
            x0, y0 = int(w_i * 0.2), int(h_i * 0.2)
            x1, y1 = int(w_i * 0.8), int(h_i * 0.8)
            center_crop = cropped_img[y0:y1, x0:x1]
            fallback_colors, fallback_hsv, fallback_method = extract_pill_colors_hsv(center_crop, contour=None, visualize=debug)
            if fallback_colors:
                return "其他", fallback_colors, fallback_hsv, f"{fallback_method}_CenterCrop"
            return "其他", [], None, "No_Contours"
            
    except Exception as e:
        return "錯誤", [], None, "Error"

def validate_color_analysis(contours, area_ratio, threshold=0.1):
    """
    Validate if color analysis should proceed based on criteria
    """
    if len(contours) > 5: # Multiple small thresholds
        return False
    if area_ratio < threshold: # Contour too small
        return False
    return True

ratios_list = []
