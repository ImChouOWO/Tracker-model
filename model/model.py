import torch
import json
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms
import os
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import random

JOSNFLODER = 'C:/Projects/python code/trackermodel/bbox_info/json'  
OUTPKL = 'C:/Projects/python code/trackermodel/bbox_info/pkl/bboxs_info.pkl'  # 指定輸出的PKL檔案名稱

# training data path
# BBOX_JSON_DIR = "C:/Projects/python code/trackermodel/bbox_info/json"
# IMG_DIR = "C:/Projects/python code/trackermodel/bbox_info/img"
# IMG_BBOX_DIR = 'C:/Projects/python code/trackermodel/bbox_info/img_bbox'

# test data path
BBOX_JSON_DIR = "C:/Projects/python code/trackermodel/test_dataset/bbox"
IMG_DIR = "C:/Projects/python code/trackermodel/test_dataset/img"
IMG_BBOX_DIR = 'C:/Projects/python code/trackermodel/test_dataset/img_bbox'


SAVE_MODEL_DIR = 'C:/Projects/python code/trackermodel/saved_models'
STATS_DIR = 'C:/Projects/python code/trackermodel/stats'
PREDICTIONS_DIR = 'C:/Projects/python code/trackermodel/predictions'

PRE_TRAINED_MODEL = "C:/Projects/python code/trackermodel/saved_models/100_epoch/best_model_weights.pth"
# PRE_TRAINED_MODEL = "C:/Projects/python code/trackermodel/saved_models/100_epoch/final_model_weights.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CreatDataset(Dataset):
    def __init__(self, img_dir, img_bbox_dir, bbox_json_dir, transform=None) -> None:
        self.img_dir = img_dir
        self.img_bbox_dir = img_bbox_dir
        self.bbox_json_dir = bbox_json_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.img_bbox_files = sorted([f for f in os.listdir(img_bbox_dir) if os.path.isfile(os.path.join(img_bbox_dir, f))])
        self.bbox_json_files = sorted([f for f in os.listdir(bbox_json_dir) if os.path.isfile(os.path.join(bbox_json_dir, f))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img_bbox_path = os.path.join(self.img_bbox_dir, self.img_bbox_files[idx])
        bbox_json_path = os.path.join(self.bbox_json_dir, self.bbox_json_files[idx])
        
        img = Image.open(img_path).convert("RGB")
        img_bbox = Image.open(img_bbox_path).convert("RGB")
        
        with open(bbox_json_path) as f:
            bbox_info = json.load(f)
        
        bbox = torch.tensor(bbox_info["bbox"], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
            img_bbox = self.transform(img_bbox)
        
        return img, img_bbox, bbox

class Train(nn.Module):
    def __init__(self, early_stop) -> None:
        super(Train, self).__init__()
        self.triple_cnn2d = TripleCNN2D()
        self.triple_cnn1d = TripleCNN1D()
        self.self_attention = SelfAttention()
        self.transformer = Transformer()
        self.fc = nn.Linear(256, 4)
        self.early_stop = early_stop
        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()])
        self.dataset = CreatDataset(IMG_DIR, IMG_BBOX_DIR, BBOX_JSON_DIR, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def forward(self, img, img_with_bbox, bbox):
        img = img.to(device)
        img_with_bbox = img_with_bbox.to(device)
        bbox = bbox.to(device)

        img_features = self.triple_cnn2d(img)
        img_bbox_features = self.triple_cnn2d(img_with_bbox)
        bbox_features = self.triple_cnn1d(bbox.unsqueeze(1))

        # 將卷積層輸出的形狀調整為 (batch_size, seq_length, feature_dim)
        img_features = img_features.view(img_features.size(0), -1, img_features.size(1))
        img_bbox_features = img_bbox_features.view(img_bbox_features.size(0), -1, img_bbox_features.size(1))
        bbox_features = bbox_features.view(bbox_features.size(0), -1, bbox_features.size(1))

        attention_out = self.self_attention(img_features, img_bbox_features, bbox_features)
        transformer_out = self.transformer(attention_out, attention_out)

        # 均值池化
        transformer_out = torch.mean(transformer_out, dim=1)

        output = self.fc(transformer_out)
        return output
    
    def train_model(self, epochs=10, patience=5, clip_value=1.0):
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        loss_function = nn.SmoothL1Loss()
        dataloader = self.dataloader

        stats = []
        predictions = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            mae = 0.0
            rmse = 0.0
            count = 0
            
            all_targets = []
            all_outputs = []

            for img, img_with_bbox, bbox in dataloader:
                optimizer.zero_grad()
                output = self.forward(img, img_with_bbox, bbox)
                target = bbox.to(device)  # 假設目標是 bbox，具體根據需求調整

                # 檢查並處理 NaN 值
                if torch.isnan(output).any():
                    print(f"NaN detected in output at epoch {epoch + 1}")
                    output = torch.where(torch.isnan(output), torch.zeros_like(output), output)

                loss = loss_function(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
                optimizer.step()
                
                epoch_loss += loss.item()
                mae += torch.sum(torch.abs(output - target)).item()
                rmse += torch.sum((output - target) ** 2).item()
                count += target.size(0)

                all_targets.append(target.cpu().detach().numpy())
                all_outputs.append(output.cpu().detach().numpy())

                # 記錄預測結果
                predictions.append({
                    'epoch': epoch + 1,
                    'target': target.cpu().detach().numpy(),
                    'output': output.cpu().detach().numpy()
                })
            
            epoch_loss /= len(dataloader)
            mae /= count
            rmse = np.sqrt(rmse / count)

            # 計算每個輸出的 F1 分數，然後取平均值
            all_targets = np.concatenate(all_targets, axis=0)
            all_outputs = np.concatenate(all_outputs, axis=0)

            # 檢查並處理 NaN 值
            if np.isnan(all_outputs).any():
                print(f"NaN detected in all_outputs at epoch {epoch + 1}")
                all_outputs = np.nan_to_num(all_outputs)

            f1_scores = []
            for i in range(all_targets.shape[1]):
                f1 = f1_score(np.round(all_targets[:, i]), np.round(all_outputs[:, i]), average='macro')
                f1_scores.append(f1)
            avg_f1 = np.mean(f1_scores)

            stats.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'mae': mae,
                'rmse': rmse,
                'f1_score': avg_f1
            })

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Avg F1 Score: {avg_f1:.4f}')

            # 早停機制
            if self.early_stop == True and epoch > 100:

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    # 儲存最佳模型權重
                    torch.save(self.state_dict(), os.path.join(SAVE_MODEL_DIR, 'best_model_weights.pth'))
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # 儲存最終模型權重
        torch.save(self.state_dict(), os.path.join(SAVE_MODEL_DIR, 'final_model_weights.pth'))

        # 儲存訓練過程中的指標
        stats_df = pd.DataFrame(stats)
        os.makedirs(STATS_DIR, exist_ok=True)
        stats_df.to_csv(os.path.join(STATS_DIR, 'training_stats.csv'), index=False)

        # 儲存預測結果
        predictions_df = pd.DataFrame(predictions)
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        predictions_df.to_csv(os.path.join(PREDICTIONS_DIR, 'predictions.csv'), index=False)

class TripleCNN2D(nn.Module):
    def __init__(self):
        super(TripleCNN2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm([64, 128, 128])  # 使用LayerNorm，形狀匹配卷積層的輸出
        self.conv1_residual = nn.Conv2d(3, 64, kernel_size=1, stride=2)  # 用於調整維度的1x1卷積

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([128, 64, 64])  # 使用LayerNorm，形狀匹配卷積層的輸出
        self.conv2_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2)  # 用於調整維度的1x1卷積

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.ln3 = nn.LayerNorm([256, 32, 32])  # 使用LayerNorm，形狀匹配卷積層的輸出
        self.conv3_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2)  # 用於調整維度的1x1卷積

    def forward(self, x):
        res = self.conv1_residual(x)
        out = self.conv1(x)
        out = self.ln1(out)
        out += res

        res = self.conv2_residual(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out += res

        res = self.conv3_residual(out)
        out = self.conv3(out)
        out = self.ln3(out)
        out += res
        
        return out

class TripleCNN1D(nn.Module):
    def __init__(self):
        super(TripleCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1)
        self.ln1 = nn.LayerNorm([64, 2])  # 使用LayerNorm，形狀匹配卷積層的輸出
        self.conv1_residual = nn.Conv1d(1, 64, kernel_size=1, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([128, 1])  # 使用LayerNorm，形狀匹配卷積層的輸出
        self.conv2_residual = nn.Conv1d(64, 128, kernel_size=1, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.ln3 = nn.LayerNorm([256, 1])  # 使用LayerNorm，形狀匹配卷積層的輸出
        self.conv3_residual = nn.Conv1d(128, 256, kernel_size=1, stride=2)

    def forward(self, x):
        
        res = self.conv1_residual(x)
        out = self.conv1(x)
        out = self.ln1(out)
        out += res
        

        res = self.conv2_residual(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out += res
        

        res = self.conv3_residual(out)
        out = self.conv3(out)
        out = self.ln3(out)
        out += res
        

        return out

class SelfAttention(nn.Module):
    def __init__(self, input_dim=256) -> None:
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.value = nn.Linear(self.input_dim, self.input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_img, x_img_bbox, x_bbox):  # x.shape (batch_size, seq_length, input_dim)
        queries = self.query(x_img)
        keys = self.key(x_img_bbox)
        values = self.value(x_bbox)

        # 確保 values 的形狀匹配
        values = values.repeat(1, queries.size(1), 1)

        # 計算注意力權重
        score = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(score)

        # 計算加權輸出
        weighted = torch.bmm(attention, values)

        return weighted

class Transformer(nn.Module):
    def __init__(self, input_dim=256, model_dim=256, num_heads=8, num_layers=6, dropout=0.01) -> None:
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, input_dim)

    def forward(self, src, tgt):  # src為輸入特徵
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc(output)
        return output
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def calculate_iou(box1, box2):
    # 計算兩個框的交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 計算兩個框的並集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # 計算 IoU
    iou = intersection_area / union_area
    return iou

class Predict:
    def __init__(self, model_path, device):
        self.model = Train(early_stop=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)  # 確保模型在正確的設備上
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def predict(self, img_path, img_bbox_path, bbox_path):
        # 加載和預處理圖像
        image = Image.open(img_path).convert('RGB')
        image_bbox = Image.open(img_bbox_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        image_bbox = self.transform(image_bbox).unsqueeze(0).to(self.device)

        # 加載邊界框信息
        with open(bbox_path) as f:
            bbox_info = json.load(f)
        bbox = torch.tensor(bbox_info["bbox"], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_bbox = self.model(image, image_bbox, bbox)

        return predicted_bbox.cpu().numpy()

    def evaluate_iou(self, predicted_bbox, next_bbox_path):
        with open(next_bbox_path) as f:
            bbox_info = json.load(f)
        true_bbox = np.array(bbox_info["bbox"])

        # 計算 IoU
        iou = calculate_iou(predicted_bbox[0], true_bbox)
        return iou

def load_dataset():
    imgs = sorted(os.listdir(IMG_DIR))
    img_bboxs = sorted(os.listdir(IMG_BBOX_DIR))
    bbox_jsons = sorted(os.listdir(BBOX_JSON_DIR))
    
    img_dir_path = IMG_DIR
    img_bbox_dir_path = IMG_BBOX_DIR
    bbox_dir_path = BBOX_JSON_DIR

    predictor = Predict(model_path=PRE_TRAINED_MODEL, device=device)
    socre_list = []
    for idx in range(len(imgs) - 1):  # 確保不會超出範圍
        # 當前檔案的預測
        predicted_bbox = predictor.predict(
            img_path=os.path.join(img_dir_path, imgs[idx]), 
            img_bbox_path=os.path.join(img_bbox_dir_path, img_bboxs[idx]), 
            bbox_path=os.path.join(bbox_dir_path, bbox_jsons[idx])
        )
        print("Predicted BBox:", predicted_bbox)
        
        # 使用當前預測與下一個真實邊界框計算 IoU
        iou_score = predictor.evaluate_iou(
            predicted_bbox, 
            next_bbox_path=os.path.join(bbox_dir_path, bbox_jsons[idx + 1])
        )
        socre_list.append(iou_score)
        print("IoU Score:", iou_score)
    print(sum(socre_list) / len(socre_list))
    
    socre_df = pd.DataFrame(socre_list)
    
    # socre_df.to_csv(os.path.join(STATS_DIR, 'training_socre.csv'), index=False)
    socre_df.to_csv(os.path.join(STATS_DIR, 'test_socre.csv'), index=False)


if __name__ == "__main__":
    # # 初始化模型
    # model = Train(early_stop=True)

    # # 訓練模型
    # model.train_model(epochs=1000)
    set_seed(40000)
    load_dataset()