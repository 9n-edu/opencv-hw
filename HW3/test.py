import cv2
import numpy as np

# --- 全域變數 ---
sample_pixels = []  # 儲存採樣的像素值
ref_frame = None

# --- 1. 滑鼠互動功能 ---
def mouse_callback(event, x, y, flags, param):
    global sample_pixels, ref_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if ref_frame is None: return
        
        # 採樣 BGR 數值
        b, g, r = ref_frame[y, x]
        sample_pixels.append([b, g, r])

        # 在畫面上標記綠色圓點
        cv2.circle(ref_frame, (x, y), 5, (0, 255, 0), -1)
        print(f"採樣: BGR({b},{g},{r}) - 累積樣本數: {len(sample_pixels)}")

# --- 2. Trackbar 空回呼函式 ---
def nothing(x):
    pass

# --- 3. 數學運算：建立高斯模型 ---
def train_gaussian_model(samples):
    # data = (N, 3) => N 個樣本，每個樣本有 3 個維度 (B,G,R)
    data = np.array(samples, dtype=np.float64)
    
    # 計算平均向量
    mean_vec = np.mean(data, axis=0)
    
    # 計算共變異數矩陣 (rowvar=False 代表每一行是一個變數 B,G,R)
    cov_matrix = np.cov(data, rowvar=False)
    
    # 計算反矩陣 (加上極小值防止奇異矩陣錯誤)
    inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(3) * 1e-6)
    
    return mean_vec, inv_cov_matrix

# --- 4. 數學運算：馬哈拉諾比斯距離切割 (核心演算法) ---
def manual_mahalanobis_segment(image, mean, inv_cov, threshold):
    """
    計算每個像素與膚色分佈中心的 '統計距離'
    threshold: 動態傳入的閾值 (標準差倍數)
    """
    img_float = image.astype(np.float64)
    h, w, c = img_float.shape
    
    # 展平像素以利矩陣運算
    pixels = img_float.reshape(-1, 3)
    
    # 減去平均值
    diff = pixels - mean
    
    # 計算馬哈拉諾比斯距離平方: (x-u)^T * Cov^-1 * (x-u)
    left_term = np.dot(diff, inv_cov)
    dist_sq = np.sum(left_term * diff, axis=1)
    
    # 閾值比較
    mask_flat = dist_sq <= (threshold ** 2)
    
    # 重塑回影像形狀
    mask = mask_flat.reshape(h, w).astype(np.uint8) * 255
    
    return mask

# --- 5. 簡單範圍切割 (RGB/HSV 通用) ---
def manual_range_segment(image, lower, upper):
    mask = np.all((image >= lower) & (image <= upper), axis=2)
    return mask.astype(np.uint8) * 255

# --- 6. 去除雜訊並保留最大色塊 ---
def remove_noise_and_keep_largest(mask):
    # 形態學開運算 (先侵蝕再膨脹) 去除小白點
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 找出輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    
    if contours:
        # 找出面積最大的輪廓
        max_cnt = max(contours, key=cv2.contourArea)
        
        # 過濾太小的雜訊 (例如面積小於 1000 像素就不顯示)
        if cv2.contourArea(max_cnt) > 1000:
            cv2.drawContours(clean_mask, [max_cnt], -1, 255, -1)
            
    return clean_mask

# --- 主程式 ---
def main():
    global ref_frame
    cap = cv2.VideoCapture(0)
    
    # === 步驟 1: 建立參考圖 ===
    print("=== 步驟 1: 建立參考圖 ===")
    print("請調整姿勢，按 's' 截取畫面")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # 鏡像翻轉，比較直覺
        cv2.imshow("Setup", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            ref_frame = frame.copy()
            break
    cv2.destroyWindow("Setup")
    
    # === 步驟 2: 採樣 ===
    print("\n=== 步驟 2: 採樣 (Modeling) ===")
    print("請點擊參考圖中的手部 (亮部、暗部、陰影都要點，約 5-10 點)")
    print("點完按 'c' 開始計算")
    
    cv2.namedWindow("Reference")
    cv2.setMouseCallback("Reference", mouse_callback)
    
    while True:
        cv2.imshow("Reference", ref_frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            if len(sample_pixels) > 2:
                break
            else:
                print("點太少了！統計模型需要更多數據 (至少 3 點)")
    cv2.destroyWindow("Reference")

    # --- 計算基礎統計值 (移出迴圈以提升效能) ---
    # 1. RGB 基礎值
    arr = np.array(sample_pixels)
    rgb_base_min = np.min(arr, axis=0)
    rgb_base_max = np.max(arr, axis=0)
    
    # 2. HSV 基礎值 (先將採樣點轉為 HSV)
    # 需增加維度變為 (1, N, 3) 才能用 cvtColor，算完再轉回 (N, 3)
    hsv_samples = cv2.cvtColor(np.array([sample_pixels], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0]
    h_base_min = np.min(hsv_samples, axis=0)
    h_base_max = np.max(hsv_samples, axis=0)

    # 3. GMM 模型訓練
    g_mean, g_inv_cov = train_gaussian_model(sample_pixels)

    # === 步驟 3: 設定控制視窗與 Trackbars ===
    cv2.namedWindow("Controls")
    cv2.resizeWindow("Controls", 400, 250)

    # 建立滑動條
    # 參數：(名稱, 視窗, 預設值, 最大值, 回呼函式)
    cv2.createTrackbar("RGB Tolerance", "Controls", 30, 100, nothing)    # RGB 寬容度
    cv2.createTrackbar("H Tolerance", "Controls", 10, 90, nothing)       # 色相 (Hue) 寬容度
    cv2.createTrackbar("SV Tolerance", "Controls", 40, 100, nothing)     # 飽和度/亮度 寬容度
    cv2.createTrackbar("GMM Thresh x10", "Controls", 35, 100, nothing)   # GMM 閾值 (數值需 /10)

    print("\n=== 步驟 3: 結果顯示 (互動模式) ===")
    print("請試著拖拉 'Controls' 視窗上的滑動條來優化效果")
    print("按 'q' 離開")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 1. 讀取 Trackbar 數值
        rgb_tol = cv2.getTrackbarPos("RGB Tolerance", "Controls")
        h_tol = cv2.getTrackbarPos("H Tolerance", "Controls")
        sv_tol = cv2.getTrackbarPos("SV Tolerance", "Controls")
        gmm_thresh_int = cv2.getTrackbarPos("GMM Thresh x10", "Controls")
        
        # 將整數轉為浮點數 (例如 35 -> 3.5)
        gmm_threshold = gmm_thresh_int / 10.0

        # --- A. RGB 動態切割 ---
        rgb_lower = np.clip(rgb_base_min - rgb_tol, 0, 255)
        rgb_upper = np.clip(rgb_base_max + rgb_tol, 0, 255)
        
        mask_rgb_raw = manual_range_segment(frame, rgb_lower, rgb_upper)
        mask_rgb = remove_noise_and_keep_largest(mask_rgb_raw)
        res_rgb = cv2.bitwise_and(frame, frame, mask=mask_rgb)
        
        # --- B. HSV 動態切割 ---
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Hue 個別處理，SV 一起處理
        hsv_lower_bound = h_base_min - [h_tol, sv_tol, sv_tol]
        hsv_upper_bound = h_base_max + [h_tol, sv_tol, sv_tol]
        
        hsv_lower = np.clip(hsv_lower_bound, 0, 255)
        hsv_upper = np.clip(hsv_upper_bound, 0, 255)
        # 修正 Hue 超出 180 的情況
        if hsv_upper[0] > 180: hsv_upper[0] = 180

        mask_hsv_raw = manual_range_segment(hsv_frame, hsv_lower, hsv_upper)
        mask_hsv = remove_noise_and_keep_largest(mask_hsv_raw)
        res_hsv = cv2.bitwise_and(frame, frame, mask=mask_hsv)
        
        # --- C. GMM 動態切割 ---
        # 傳入動態調整後的 threshold
        mask_gmm_raw = manual_mahalanobis_segment(frame, g_mean, g_inv_cov, threshold=gmm_threshold)
        mask_gmm = remove_noise_and_keep_largest(mask_gmm_raw)
        res_gmm = cv2.bitwise_and(frame, frame, mask=mask_gmm)

        # --- 顯示資訊與結果 ---
        # 在原圖上顯示目前的 GMM 閾值
        info_text = f"GMM Threshold: {gmm_threshold}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("0. Original", frame)
        cv2.imshow("1. RGB (Tunable)", res_rgb)
        cv2.imshow("2. HSV (Tunable)", res_hsv)
        cv2.imshow("3. GMM (Tunable)", res_gmm)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()