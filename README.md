## 簡介/Introduction
 💡 **當前專案尚未完成，仍有許多需要改進的地方，如資料集、模型結構等**
 
本專案遵循以下工作流程進行研究

>[!NOTE]
>![work flow](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/work%20flowdrawio.drawio.png)
>
>本專案目前利用以下三項資料構建輸入
>
> - 原始圖片
> - 包含偵測框的圖片
> - 偵測框資訊
>
>
>| Data Set    | IMG            | IMG and BBOX   | BBOX           |
>|-------------|----------------|----------------|----------------|
>| 數量        | 33             | 33             | 33             |
>| 資料型別    | 256X256 RGB    | 256X256 RGB    | Float          |
> ---
>![model structure](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/trackermodel.jpg)
>
>
>![model structure_block](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/trackermodel_block.drawio.png)
---
## 績效
>[!NOTE]
>![LOSS](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/Loss.png)
>![MAE](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/MAE.png)

## 當前硬體

| Device      | GPU            | RAM            | SYSTEM         |
|-------------|----------------|----------------|----------------|
|             | 3060 12G       | 32G            | Windows 11     |

## 待改進

> [!IMPORTANT]
> 
> - [x] 擴增資料集
> - [x] 完善輸入特徵
> - [x] 完善模型結構
> - [x] 改善YOLOV8權重 
> - [x] 比較base line

