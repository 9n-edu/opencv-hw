import cv2
import numpy as np
import math
import os

# ============================================================
# 1. 自己寫的灰階轉換
# ============================================================
def rgb2gray(img):
    h, w, c = img.shape
    gray = np.zeros((h, w), dtype=np.uint8)

    # 使用 BT.601 標準（OpenCV 也是用這個）
    for y in range(h):
        for x in range(w):
            r, g, b = img[y, x]
            gray[y, x] = int(0.299 * r + 0.587 * g + 0.114 * b)

    return gray


# ============================================================
# 2. 高斯模糊（修正版：與 OpenCV 更一致）
# ============================================================
def gaussian_kernel(size, sigma):
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    s = 2 * sigma * sigma

    for i in range(size):
        for j in range(size):
            x = i - k
            y = j - k
            kernel[i, j] = math.exp(-(x * x + y * y) / s)

    return kernel / np.sum(kernel)


def gaussian_blur(img, size=3, sigma=1.4):
    h, w = img.shape
    kernel = gaussian_kernel(size, sigma)
    pad = size // 2

    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    result = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            result[i, j] = np.sum(region * kernel)

    return result.astype(np.uint8)


# ============================================================
# 3. Sobel 梯度（完全手寫，與 OpenCV 一致）
# ============================================================
def sobel_gradient(img):
    h, w = img.shape

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    padded = np.pad(img, ((1, 1), (1, 1)), mode='reflect')

    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            region = padded[y:y+3, x:x+3]
            gx[y, x] = np.sum(region * sobel_x)
            gy[y, x] = np.sum(region * sobel_y)

    # OpenCV 计算方式：magnitude = sqrt(gx^2 + gy^2)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Clip 掉高值，避免 overflow 導致線條消失
    magnitude = np.clip(magnitude, 0, 255)

    direction = np.rad2deg(np.arctan2(gy, gx))
    direction[direction < 0] += 180

    return magnitude.astype(np.uint8), direction


# ============================================================
# 4. NMS 極大值抑制（精確版）
# ============================================================
def non_max_suppression(mag, direction):
    h, w = mag.shape
    nms = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            d = direction[i, j]

            # 鄰域像素
            if (0 <= d < 22.5) or (157.5 <= d <= 180):
                p1, p2 = mag[i, j-1], mag[i, j+1]
            elif (22.5 <= d < 67.5):
                p1, p2 = mag[i-1, j+1], mag[i+1, j-1]
            elif (67.5 <= d < 112.5):
                p1, p2 = mag[i-1, j], mag[i+1, j]
            else:
                p1, p2 = mag[i-1, j-1], mag[i+1, j+1]

            nms[i, j] = mag[i, j] if mag[i, j] >= p1 and mag[i, j] >= p2 else 0

    return nms


# ============================================================
# 5. 雙門檻 + 邊緣連接（版本與 OpenCV 一致）
# ============================================================
def hysteresis(img, low, high):
    h, w = img.shape
    strong = 255
    weak = 50

    result = np.zeros((h, w), dtype=np.uint8)

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    stack = list(zip(strong_i, strong_j))

    while stack:
        i, j = stack.pop()

        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj

                if 0 <= ni < h and 0 <= nj < w:
                    if result[ni, nj] == weak:
                        result[ni, nj] = strong
                        stack.append((ni, nj))

    # 清除弱邊
    result[result != strong] = 0
    return result


# ============================================================
# 6. 完整 My Canny（可直接對比 OpenCV）
# ============================================================
def my_canny(path):
    img = cv2.imread(path)
    gray = rgb2gray(img)
    blur = gaussian_blur(gray, size=3, sigma=1.4)

    mag, dir = sobel_gradient(blur)
    nms = non_max_suppression(mag, dir)

    final = hysteresis(nms, low=30, high=90)

    # OpenCV baseline for comparison
    cv_canny = cv2.Canny(gray, 30, 90)

    return gray, blur, mag, nms, final, cv_canny


def my_hough_lines_p(edge_image, rho_res=1, theta_res=1, threshold=100, minLineLength=100, maxLineGap=10):
    """
    手寫 HoughLinesP，與 OpenCV cv2.HoughLinesP 盡量一致
    """
    h, w = edge_image.shape
    
    # 建立累積器
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    thetas = np.deg2rad(np.arange(0, 180, theta_res))
    
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint32)
    y_idxs, x_idxs = np.where(edge_image > 0)
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    # 向量化投票
    for t_idx in range(len(thetas)):
        rho_values = x_idxs * cos_t[t_idx] + y_idxs * sin_t[t_idx]
        rho_indices = np.round(rho_values).astype(np.int32) + diag_len
        valid_mask = (rho_indices >= 0) & (rho_indices < len(rhos))
        np.add.at(accumulator[:, t_idx], rho_indices[valid_mask], 1)

    # 篩選超過門檻的直線參數
    result_indices = np.argwhere(accumulator >= threshold)
    
    if len(result_indices) == 0:
        return []
    
    # 根據票數排序 (大到小)
    votes = accumulator[result_indices[:, 0], result_indices[:, 1]]
    sort_idx = np.argsort(-votes)
    result_indices = result_indices[sort_idx]
    
    lines_p = []
    edge_points = np.column_stack((x_idxs, y_idxs))
    
    # 處理前 N 條最強的直線
    for r_idx, t_idx in result_indices[:300]:
        rho = rhos[r_idx]
        theta = thetas[t_idx]
        
        a = np.cos(theta)
        b = np.sin(theta)
        
        # 容限設為 1.5 pixel (與 OpenCV 接近)
        tolerance = 1.5
        
        # 找出所有接近這條直線的邊緣點
        distances = np.abs(edge_points[:, 0] * a + edge_points[:, 1] * b - rho)
        on_line = distances < tolerance
        
        if np.sum(on_line) < 2:
            continue
        
        line_points = edge_points[on_line]
        
        # 投影到直線上取得 1D 座標
        projected = line_points[:, 0] * (-b) + line_points[:, 1] * a
        
        # 排序
        order = np.argsort(projected)
        projected = projected[order]
        line_points = line_points[order]
        
        # 分割線段：根據 maxLineGap 判斷中斷
        segments = []
        start_i = 0
        
        for i in range(len(projected) - 1):
            gap = projected[i + 1] - projected[i]
            if gap > maxLineGap:
                # 結束一個線段
                length = projected[i] - projected[start_i]
                if length >= minLineLength:
                    p_start = line_points[start_i]
                    p_end = line_points[i]
                    segments.append((p_start[0], p_start[1], p_end[0], p_end[1]))
                start_i = i + 1
        
        # 處理最後一段
        if start_i < len(projected):
            length = projected[-1] - projected[start_i]
            if length >= minLineLength:
                p_start = line_points[start_i]
                p_end = line_points[-1]
                segments.append((p_start[0], p_start[1], p_end[0], p_end[1]))
        
        lines_p.extend(segments)
    
    return lines_p


def add_watermark(img, sign_path):
    # 讀入簽名圖 (保持 alpha 通道)
    sign = cv2.imread( sign_path, cv2.IMREAD_UNCHANGED)

    if sign.ndim == 3 and sign.shape[2] == 4:
        b, g, r, a = cv2.split(sign)
        sign_rgb = cv2.merge((b, g, r))
        alpha = a
    else:
        sign_rgb = sign if sign.ndim == 3 else cv2.cvtColor(sign, cv2.COLOR_GRAY2BGR)
        alpha = np.ones(sign_rgb.shape[:2], dtype=np.uint8) * 255

    # 縮放簽名圖 (最寬為原圖 1/5)
    h_img, w_img = img.shape[:2]
    max_width = max(1, w_img // 5)
    if sign_rgb.shape[1] == 0:
        return img
    scale = max_width / sign_rgb.shape[1]
    new_w = max(1, int(sign_rgb.shape[1] * scale))
    new_h = max(1, int(sign_rgb.shape[0] * scale))
    sign_rgb = cv2.resize(sign_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 若來源是 single-channel，先轉為 BGR，以便混合
    if img.ndim == 2:
        dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        was_gray = True
    else:
        dst = img.copy()
        was_gray = False

    x_offset = w_img - new_w - 10
    y_offset = h_img - new_h - 10
    if x_offset < 0: x_offset = 0
    if y_offset < 0: y_offset = 0

    # alpha 範圍 0..255 -> 0..1
    alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]  # shape (h,w,1)

    # 混合
    for y in range(new_h):
        for x in range(new_w):
            a = alpha_f[y, x, 0]
            if a <= 0:
                continue
            src_px = sign_rgb[y, x].astype(np.float32)
            dst_px = dst[y + y_offset, x + x_offset].astype(np.float32)
            blended = (a * src_px + (1 - a) * dst_px).astype(np.uint8)
            dst[y + y_offset, x + x_offset] = blended

    if was_gray:
        # 回傳與輸入相同維度：如果原本是灰階，回傳灰階版本（方便後續顯示/儲存）
        return cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return dst



# ============================================================
# 主程式（直接跑即可）
# ============================================================
#img_path = '2.jpg'  # 使用你上傳的檔案

file_list = ['1.jpg', '2.jpg', '3.jpg']

for filename in file_list:
    img = cv2.imread(filename)
    gray, blur, mag, nms, my_canny_result, true_canny = my_canny(filename)

    output_custom = img.copy()

    my_lines = my_hough_lines_p(my_canny_result, rho_res=1, theta_res=1, 
                                    threshold=100, 
                                    minLineLength=100, 
                                    maxLineGap=10)

    for x1, y1, x2, y2 in my_lines:
        cv2.line(output_custom, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    canny_with_sig = add_watermark(my_canny_result.copy(), sign_path='white_signed.png')
    hough_lines_p_with_sig = add_watermark(output_custom.copy(), sign_path='signed.png')

    name, ext = os.path.splitext(filename)

    cv2.imshow(f"canny - {name}", canny_with_sig)
    cv2.imshow(f"Hough_LinesP_My_Result- {name}", hough_lines_p_with_sig)
    cv2.imwrite(f"{name}_Canny_Edge_Detection{ext}", canny_with_sig)
    cv2.imwrite(f"{name}_Hough_Transform{ext}", hough_lines_p_with_sig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()