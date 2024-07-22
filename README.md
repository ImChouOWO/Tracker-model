## 簡介/Introduction
 💡 **當前專案尚未完成，仍有許多需要改進的地方，如資料集、模型結構等**
 > **The current project is not yet completed and still has many issue that need improvement, such as the dataset, model structure,etc.**
 
### 本專案遵循以下工作流程進行研究
>This project follows the research workflow outlined below.


![work flow](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/work%20flowdrawio.drawio.png)


### 本專案目前利用以下三項資料構建輸入
 > This project currently utilizes the following three types of data as inputs

>[!NOTE]
>
> - 原始圖片(Original images)
> - 包含偵測框的圖片(Images with bounding boxes)
> - 偵測框資訊(Bounding box information)
>
>
>| Data Set    | IMG            | IMG and BBOX   | BBOX           |
>|-------------|----------------|----------------|----------------|
>| 數量        | 33             | 33             | 33             |
>| 資料型別    | 256X256 RGB    | 256X256 RGB    | Float          |
>
>| Data Set    | IMG            | IMG and BBOX   | BBOX           |
>|-------------|----------------|----------------|----------------|
>| Quantity    | 33             | 33             | 33             |
>| Data Type   | 256X256 RGB    | 256X256 RGB    | Float          |
> ---
### 主架構/main strucure
![model structure](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/trackermodel.jpg)

### CNN區塊/Triple CNN
![model structure_block](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/trackermodel_block.drawio.png)
---
## 績效/Performance


![LOSS](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/Loss.png)
![MAE](https://github.com/ImChouOWO/Tracker-model/blob/main/structure/MAE.png)


 ## IOU Score
$$
IOU = \frac{Area\ of\ Overlap}{Area\ of\ Union}
$$

- **Overlap**：預測邊界框與實際邊界框重疊的區域面積。
  >**Area of Overlap**: The area where the predicted bounding box and the ground truth bounding box overlap.
- **Union**：預測邊界框和實際邊界框覆蓋的總區域面積。
  >**Area of Union**: The total area covered by the predicted bounding box and the ground truth bounding box.

---

|                | Best      | Average   |
|----------------|-----------|-----------|
| 內部資料       | 0.908486  | 0.593728  |
| 外部資料       | 0.335145  | 0.11712   |

|                | Best      | Average   |
|----------------|-----------|-----------|
| Internal Data  | 0.908486  | 0.593728  |
| External Data  | 0.335145  | 0.11712   |


## 當前硬體/Current Hardware

| Device      | GPU            | RAM            | SYSTEM         |
|-------------|----------------|----------------|----------------|
|             | 3060 12G       | 32G            | Windows 11     |

## 待改進/To Be Improved

> [!IMPORTANT]
> 
> - [x] 擴增資料集 (Expand dataset)
> - [x] 完善輸入特徵 (Improve input feature)
> - [x] 完善模型結構 (Improve model strucure)
> - [x] 改善YOLOV8權重 (Optimize YoloV8 weight)
> - [x] 比較base line (Compare with base line)

