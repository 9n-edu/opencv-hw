import cv2
import numpy as np
import math

img = cv2.imread('cat.jpg')

#照片轉黑白
gray = np.zeros(img.shape[:2], dtype=np.uint8)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        b, g, r = img[y, x]
        gray_value = int(0.114 * b + 0.587 * g + 0.299 * r)
        gray[y, x] = gray_value


# 放大/縮小
h, w, c = img.shape
print("原始尺寸: ")
print(h, w)
scale = float(input("輸入放大或縮小幾倍 (0.1~2): "))

new_h = int(h * scale)
new_w = int(w * scale)
print("新尺寸: ")
print(new_h, new_w)

#最近鄰
nearest = np.zeros((new_h, new_w, c), dtype=img.dtype)

for y in range(new_h):
    for x in range(new_w):
        src_y = int(y / scale)
        src_x = int(x / scale)

        #保護邊界
        src_x  = min(src_x, w - 1)
        src_y  = min(src_y, h - 1)
        nearest[y, x] = img[src_y, src_x]

# 線性放大/縮小


bilinear = np.zeros((new_h, new_w, c), dtype=img.dtype) # 建立新影像

# 雙重迴圈遍歷新影像的每個像素
for y in range(new_h):
    for x in range(new_w):
        src_x = x / scale
        src_y = y / scale

        #取得左上角的像素座標
        y0 = int(src_y) 
        x0 = int(src_x) 
        # 取得右下角的像素座標
        y1 = min(y0 + 1, h - 1) 
        x1 = min(x0 + 1, w - 1)

        # 計算插值權重 
        dy = src_y - y0
        dx = src_x - x0

        # 取得四個鄰近像素的值
        Q11 = img[y0, x0]
        Q21 = img[y0, x1]
        Q12 = img[y1, x0]
        Q22 = img[y1, x1]

        # 執行雙線性插值
        top = (1 - dx) * Q11 + dx * Q21
        bottom = (1 - dx) * Q12 + dx * Q22
        pixel = (1 - dy) * top + dy * bottom

        bilinear[y, x] = np.clip(pixel, 0 , 255) #確保像素值在有效範圍內

#旋轉
angle = float(input("輸入角度: "))

theta = math.radians(angle)
cx = w/2
cy = h/2

rotated = np.zeros_like(img)

for new_y in range(h):
    for new_x in range(w):
        x = (new_x - cx) * math.cos(theta) + (new_y - cy) * math.sin(theta) + cx
        y = -(new_x - cx) * math.sin(theta) + (new_y - cy) * math.cos(theta) + cy

        if 0 <= x < w and 0 <= y < h:
            rotated[new_y, new_x] = img[int(y), int(x)]

# 讀入簽名圖 (保持 alpha 通道)
sign = cv2.imread('signed.png', cv2.IMREAD_UNCHANGED)

# 假設簽名圖是 RGBA
if sign.shape[2] == 4:
    b, g, r, a = cv2.split(sign)
    sign_rgb = cv2.merge((b, g, r))
    alpha = a
else:
    # 若沒有透明通道，就直接用 RGB
    sign_rgb = sign
    alpha = np.ones(sign_rgb.shape[:2], dtype=np.uint8) * 255

# 縮放簽名圖
max_width = img.shape[1] // 5
scale = max_width / sign_rgb.shape[1]
new_w = int(sign_rgb.shape[1] * scale)
new_h = int(sign_rgb.shape[0] * scale)
sign_rgb = cv2.resize(sign_rgb, (new_w, new_h))
alpha = cv2.resize(alpha, (new_w, new_h))

# 計算右下角位置
h, w = img.shape[:2]
x_offset = w - new_w - 10
y_offset = h - new_h - 10

# ======= 手動像素判斷疊合 =======
for y in range(new_h):
    for x in range(new_w):
        # 若 alpha 非 0 (代表該點有簽名)
        if alpha[y, x] > 0:  
            img[y + y_offset, x + x_offset] = sign_rgb[y, x]


# 顯示影像
cv2.imshow('Gray Image', gray)
cv2.imshow('nearest Image', nearest)
cv2.imshow('bilinear Image', bilinear)     
cv2.imshow('rotated Image', rotated)
cv2.imshow("Signed", img)

cv2.imwrite('gray.bmp', gray) #儲存黑白影像
cv2.imwrite('nearest.bmp', nearest) #最近鄰 
cv2.imwrite('bilinear.bmp', bilinear) #線性
cv2.imwrite('rotated.bmp', rotated) #旋轉
cv2.imwrite("add_signed.bmp", img)
cv2.waitKey(0)



