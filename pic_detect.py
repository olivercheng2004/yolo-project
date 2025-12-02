import os
import cv2
import numpy as np
import sys
from ultralytics import YOLO

# ===============================
# 取得 EXE 或開發環境下的正確根目錄
# ===============================


def get_app_dir():
    """取得應用程式執行目錄（支援 EXE + 開發模式）"""
    if getattr(sys, 'frozen', False):
        # EXE 模式：取 start.exe 所在資料夾
        return os.path.dirname(sys.executable)
    else:
        # 開發模式
        return os.path.dirname(os.path.abspath(__file__))


APP_DIR = get_app_dir()


def resource_path(relative_path):
    """取得完整路徑"""
    return os.path.join(APP_DIR, relative_path)


# ===============================
# 載入 YOLO 模型（best.pt 必須放在 EXE 同資料夾）
# ===============================
MODEL_PATH = resource_path("best.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")
yolo = YOLO(MODEL_PATH)

# ===============================
# ROI（紅框）定義：斑馬線區域
# ===============================
roi_polygon = np.array([
    [747, 381],
    [1160, 390],
    [1115, 966],
    [253, 907]
], dtype=np.int32)
ROI_TOP = roi_polygon[:, 1].min()

# ===============================
# 等待區界線（水平線位置）
# ===============================
WAIT_LINE_Y = 186  # (x, 186) 代表整條水平線

# ===============================
# 單張圖片偵測函數
# ===============================


def detect_image(img_path):
    """
    辨識單張圖片：
    - 使用 YOLO 偵測行人
    - 判斷是否在 ROI 上方（等待區）
    - 回傳在等待區的人數
    """
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERROR] 無法讀取 {img_path}")
        return 0

    # YOLO 偵測
    results = yolo(frame, imgsz=1280, conf=0.25, verbose=False)[0]
    people_waiting = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]

        if label.lower() == "pedestrian":
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 判斷是否在等待區
            if y2 < ROI_TOP and y1 > WAIT_LINE_Y:
                people_waiting += 1
                color = (0, 255, 0)  # 綠色框
            elif y2 <= WAIT_LINE_Y:
                color = (255, 255, 0)  # 黃色框
            else:
                color = (0, 0, 255)  # 紅色框

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ===============================
    # 儲存結果到 results 資料夾
    # ===============================
    result_dir = resource_path("results")
    os.makedirs(result_dir, exist_ok=True)

    out_path = os.path.join(result_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, frame)

    print(f"[RESULT] {img_path} → {people_waiting} 人在等待區 (ROI 上方)")
    return people_waiting
