import torch
from ultralytics import YOLO
import cv2
import json
import os
from tqdm import tqdm

# 設置路徑
MODEL = "C:/Projects/python code/trackermodel/yolov8pt/best.pt"
IMG = "C:/Projects/python code/trackermodel/img"

# SAVE = "C:/Projects/python code/trackermodel/bbox_info/json"
# ORIGINALS = "C:/Projects/python code/trackermodel/bbox_info/img"
# IMGBBOX = 'C:/Projects/python code/trackermodel/bbox_info/img_bbox'

SAVE = "C:/Projects/python code/trackermodel/test_dataset/bbox"
ORIGINALS = "C:/Projects/python code/trackermodel/test_dataset/img"
IMGBBOX = 'C:/Projects/python code/trackermodel/test_dataset/img_bbox'

# 創建保存文件夾
if not os.path.exists(ORIGINALS):
    os.makedirs(ORIGINALS)
if not os.path.exists(IMGBBOX):
    os.makedirs(IMGBBOX)
if not os.path.exists(SAVE):
    os.makedirs(SAVE)

def get_file_names(img_path):
    # 獲取資料夾中的所有檔案名稱，並在提取過程中顯示進度條
    file_names = [f for f in tqdm(os.listdir(img_path), desc="Reading file names") if os.path.isfile(os.path.join(img_path, f))]
    return file_names

def save_info(model_path, img_path, img_name, save_path, img_id):
    # 加載YOLOv8模型
    model = YOLO(model_path)

    # 加載圖像
    image_path = os.path.join(img_path, img_name)
    image = cv2.imread(image_path)
    
    # 進行檢測
    results = model(image)

    # 提取檢測框並保存資訊
    for idx, result in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        confidence = float(result.conf[0])
        class_id = int(result.cls[0])
        
        detection = {
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'class_id': class_id
        }
        
        # 保存單個檢測框到JSON文件
        output_json_path = os.path.join(save_path, f"bbox_info_{img_id}_{idx}.json")
        with open(output_json_path, 'w') as f:
            json.dump(detection, f, indent=4)
        
        # 保存原圖
        original_image_path = os.path.join(ORIGINALS, f"{img_id}_{idx}_{img_name}")
        cv2.imwrite(original_image_path, image)
        
        # 繪製標記框
        img_copy = image.copy()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_copy, f"ID: {class_id} Conf: {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # 保存帶有標記框的圖像
        output_image_path = os.path.join(IMGBBOX, f"{img_id}_{idx}_{img_name}")
        cv2.imwrite(output_image_path, img_copy)

def main():
    file_names = get_file_names(img_path=IMG)
    for id in tqdm(range(len(file_names)), desc="Processing img"):
        save_info(model_path=MODEL, img_path=IMG, img_name=file_names[id], save_path=SAVE, img_id=id)

if __name__ == "__main__":
    main()