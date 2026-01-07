'''
#使用內建函數
import cv2
import time

img = cv2.imread("Selfie.jpg")
x, y, w, h = map(int, input().split())

start_time = time.time()

roi = img[y:y+h, x:x+w] # img[y方向, x方向]
cv2. imshow("new" , roi)
end_time = time.time()
execution_time = end_time - start_time
print("程式執行時間：", execution_time, "秒")

cv2.waitKey()

cv2.imwrite("new.jpg", roi)
'''


#只能使用pixel的讀取和寫入，禁止使用所有陣列相關操作和內建函數，只允許一個一個點取值/給值(配合迴圈)
import cv2
import numpy as np
import time

img = cv2.imread("Selfie.jpg")

x, y, w, h = map(int, input().split())

start_time = time.time()

h = min(h, img.shape[0] - y)
w = min(w, img.shape[1] - x)

# 建立一張新的空白 ROI (h 高, w 寬, 3 通道)100 
roi = np.zeros((h, w, 3), dtype=img.dtype)

# 逐像素複製 ROI
for i in range(h):          # ROI 高度範圍
    for j in range(w):      # ROI 寬度範圍
        roi[i, j] = img[y + i, x + j]

cv2. imshow("new2" , roi)

end_time = time.time()
execution_time = end_time - start_time
print("程式執行時間：", execution_time, "秒")

cv2.waitKey()

cv2.imwrite("new2.jpg", roi)

