import cv2
import numpy as np

# 全域變數
sample_pixels = [] # 儲存採樣的像素值
ref_frame = None

# 滑鼠互動
def mouse_callback(event, x, y, flags, param):
    global sample_pixels, ref_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if ref_frame is None: return
        
        # 採樣 BGR 數值
        b, g, r = ref_frame[y, x]
        sample_pixels.append([b, g, r])

        cv2.circle(ref_frame, (x, y), 5, (0, 255, 0), -1)
        print(f"採樣: BGR({b},{g},{r}) - 累積樣本數: {len(sample_pixels)}")

# 建立高斯模型 (含共變異數)
def train_gaussian_model(samples):
    # 把 samples 轉成 NumPy 陣列，並且用 float64 型態
    # data = (N, 3) => N 個樣本，每個樣本有 3 個維度 (B,G,R)
    data = np.array(samples, dtype=np.float64)
    
    # 1. 計算平均向量
    # 計算 BGR 各自平均值
    # 計算結果 => mean_vec 是 一群顏色的中心位置
    mean_vec = np.mean(data, axis=0)
    
    # 2. 計算共變異數矩陣 (3x3)
    # rowvar=False 代表每一行是一個變數(B,G,R)，每一列是一個樣本
    # BGR 間 是否一起變化
    # rowvar=False 表示 一列一個樣本 
    cov_matrix = np.cov(data, rowvar=False)
    
    # 3. 預先計算反矩陣 (Inverse Covariance)，加速後續距離運算
    # 加上一個極小值 eye * 1e-6 防止矩陣不可逆 (Singular Matrix) 因為 樣本太少或太集中
    inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(3) * 1e-6)
    
    return mean_vec, inv_cov_matrix


# 2. 馬哈拉諾比斯距離 (Mahalanobis Distance)
# (原始影像, 平均向量, 反共變異數矩陣, 閾值)
def manual_mahalanobis_segment(image, mean, inv_cov, threshold=3.0):
    """
    計算每個像素與膚色分佈中心的 '統計距離'
    公式: D^2 = (x - mean)^T * Cov^-1 * (x - mean)
    """
    # 轉成 float 避免運算溢位
    img_float = image.astype(np.float64)
    h, w, c = img_float.shape
    
    # 1. 將影像展平成 (N, 3) 的陣列，方便矩陣運算
    # N = h * w
    # 3 是 BGR 三個通道
    pixels = img_float.reshape(-1, 3)
    
    # 2. 減去平均值 (x - mean)
    # 看像素離平均顏色多遠
    diff = pixels - mean
    
    # 3. 矩陣運算核心 (利用 dot product 加速)
    # left_term = (x-mean) * Cov^-1
    # 根據膚色分布形狀 調整距離方向、權重計算
    left_term = np.dot(diff, inv_cov)
    
    # 4. 計算距離平方 dist_sq = sum(left_term * diff)
    # left_term * diff  => 每個像素作對應相乘
    # np.sum(left_term * diff, axis=1) 一個像素 一個距離 => 所以 BGR 三個數要相加
    # (x-mean)^T * Cov^-1 * (x-mean)
    dist_sq = np.sum(left_term * diff, axis=1)
    
    # 5. 比較閾值 (Thresholding)
    # 距離越小代表越像膚色。 threshold^2 是因為我們算的是距離平方
    mask_flat = dist_sq <= (threshold ** 2)
    
    # 6. 重塑回影像形狀 (H, W)
    # reshape 回原本影像大小
    # astype  T / F => 1 / 0
    # *255 變成標準影像
    mask = mask_flat.reshape(h, w).astype(np.uint8) * 255
    
    return mask


# 3. 簡單範圍切割 (RGB/HSV通用)
def manual_range_segment(image, lower, upper):
    # 建立遮罩
    mask = np.all((image >= lower) & (image <= upper), axis=2)
    return mask.astype(np.uint8) * 255


# 去除雜訊 (保留最大色塊)

def remove_noise_and_keep_largest(mask):
    """
    這是一個清理步驟，可以有效移除背景的碎雜訊
    """
    # 1. 形態學運算：先侵蝕掉小白點，再膨脹回來 (Opening)
    # 建立一個5x5結構元素
    # 對遮罩做開運算( 先侵蝕 再膨脹 )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. 找出所有輪廓
    # 從遮罩中找出所有白色區塊邊界
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 建立全黑的遮罩
    # 只把通過條件的那塊畫上去
    clean_mask = np.zeros_like(mask)
    
    if contours:
        # 3. 找到面積最大的輪廓 (假設手是畫面中最大的膚色區塊)
        max_cnt = max(contours, key=cv2.contourArea)
        
        # 過濾太小的誤判 (例如面積小於 1000 像素就不算)
        if cv2.contourArea(max_cnt) > 1000:
            cv2.drawContours(clean_mask, [max_cnt], -1, 255, -1)
            
    return clean_mask



# 主程式

def main():
    # 宣告 ref_frame 為全域變數
    global ref_frame
    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    
    print("=== 步驟 1: 建立參考圖 ===")
    print("請按 's' 截取畫面")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        cv2.imshow("Setup", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 儲存 "參考影像" 跳出迴圈
            ref_frame = frame.copy()
            break
    cv2.destroyWindow("Setup")
    
    print("\n=== 步驟 2: 採樣 (Modeling) ===")
    print("請點擊參考圖中的手部 (亮部、暗部、陰影都要點，約 5-10 點)")
    print("點完按 'c' 開始計算")
    
    cv2.namedWindow("Reference")
    cv2.setMouseCallback("Reference", mouse_callback)
    
    while True:
        # 顯示參考影像，等待滑鼠點擊
        cv2.imshow("Reference", ref_frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            if len(sample_pixels) > 2: # 至少要3點才能算共變異數
                break
            else:
                print("點太少了！統計模型需要更多數據")
    cv2.destroyWindow("Reference")

    # --- 計算模型參數 ---
    # 1. RGB
    arr = np.array(sample_pixels)   # list 轉成 NumPy 陣列
    # 寬容度 +/- 30
    rgb_min = np.min(arr, axis=0) - 30
    rgb_max = np.max(arr, axis=0) + 30
    # 防止超過RGB合法範圍 0-255
    rgb_lower = np.clip(rgb_min, 0, 255)
    rgb_upper = np.clip(rgb_max, 0, 255)

    # 2. HSV
    # 把取樣點 轉到 HSV 空間
    hsv_samples = cv2.cvtColor(np.array([sample_pixels], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0]
    # 各通道寬容度
    h_min = np.min(hsv_samples, axis=0) - [10, 40, 40]
    h_max = np.max(hsv_samples, axis=0) + [10, 40, 40]
    # 防止超過 HSV 合法範圍
    hsv_lower = np.clip(h_min, 0, 255)
    hsv_upper = np.clip(h_max, 0, 255)
    # HSV 的 H 通道範圍是 0-180
    if hsv_upper[0] > 180: hsv_upper[0] = 180

    # 3. GMM
    # 計算 Mean 和 Inverse Covariance Matrix
    g_mean, g_inv_cov = train_gaussian_model(sample_pixels)

    print("\n=== 步驟 3: 結果顯示 ===")
    print("按 'q' 離開")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # 1. RGB 切割
        mask_rgb_raw = manual_range_segment(frame, rgb_lower, rgb_upper)
        mask_rgb = remove_noise_and_keep_largest(mask_rgb_raw)
        # 利用遮罩 把原圖切割出來
        # 只保留符合 RGB 範圍的像素，其餘黑掉
        res_rgb = cv2.bitwise_and(frame, frame, mask=mask_rgb)
        
        # 2. HSV 切割
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv_raw = manual_range_segment(hsv_frame, hsv_lower, hsv_upper)
        mask_hsv = remove_noise_and_keep_largest(mask_hsv_raw)
        res_hsv = cv2.bitwise_and(frame, frame, mask=mask_hsv)
        
        # 3. GMM 切割
        mask_gmm_raw = manual_mahalanobis_segment(frame, g_mean, g_inv_cov, threshold=3.5)
        mask_gmm = remove_noise_and_keep_largest(mask_gmm_raw)
        res_gmm = cv2.bitwise_and(frame, frame, mask=mask_gmm)

        cv2.imshow("0. Original", frame)
        cv2.imshow("1. RGB", res_rgb)
        cv2.imshow("2. HSV", res_hsv)
        cv2.imshow("3. GMM", res_gmm)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()