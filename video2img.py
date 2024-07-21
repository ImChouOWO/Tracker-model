import cv2
import os

# 設置影片路徑和保存圖像的資料夾
video_path = 'C:\\Projects\\python code\\trackermodel\\video\\video_edit.mp4'
output_folder = 'C:\\Projects\\python code\\trackermodel\\img'

# 確保保存圖像的資料夾存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打開影片
cap = cv2.VideoCapture(video_path)

frame_count = 0
save_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 每10幀保存一張圖像
    if frame_count % 10 == 0:
        save_path = os.path.join(output_folder, f'frame_{save_count:04d}.jpg')
        cv2.imwrite(save_path, frame)
        save_count += 1
    
    frame_count += 1

cap.release()
print(f'Total {save_count} images saved to {output_folder}')
