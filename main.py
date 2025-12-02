from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import glob
import os
import sqlite3
import json
from pic_detect import detect_image, get_app_dir
import base64

app = Flask(__name__)
CORS(app)

# ==============================
# 路徑統一
# ==============================
APP_DIR = get_app_dir()  # EXE 或開發模式
UPLOAD_DIR = os.path.join(APP_DIR, "uploaded_images")
RESULTS_DIR = os.path.join(APP_DIR, "results")
DB_FILE = os.path.join(APP_DIR, "detections.db")
CONTROL_FILE = os.path.join(APP_DIR, "control_signal.txt")

# ==============================
# 初始化 SQLite
# ==============================


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            people_count INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_result(filename, count):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO results (filename, people_count) VALUES (?, ?)", (filename, count))
    conn.commit()
    conn.close()


def get_latest_counts(n=3):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT people_count FROM results ORDER BY timestamp DESC LIMIT ?", (n,))
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    return rows

# ==============================
# Flask API
# ==============================


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return "No file received", 400
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)
    return f"Saved {file.filename}", 200


@app.route("/trigger", methods=["POST"])
def trigger_analysis():
    try:
        num_images = int(request.form.get("count", 3))
        thread = threading.Thread(
            target=process_latest_images, args=(num_images,))
        thread.start()
        return jsonify({"status": "ok", "msg": f"開始處理最近{num_images}張圖片"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


@app.route("/get_time", methods=["GET"])
def get_time():
    try:
        # 讀取 extend_seconds
        with open(CONTROL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        extend_seconds = data.get("extend_seconds", None)

        # 讀取 results 內的三張圖片
        image_paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jpg")),
                             key=os.path.getmtime, reverse=True)[:3]

        images_base64 = []
        for p in image_paths:
            with open(p, "rb") as img_f:
                encoded = base64.b64encode(img_f.read()).decode("utf-8")
                images_base64.append("data:image/jpeg;base64," + encoded)

        return jsonify({
            "extend_seconds": extend_seconds,
            "images": images_base64
        })

    except FileNotFoundError:
        return jsonify({"error": f"{CONTROL_FILE} not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# 背景任務
# ==============================


def process_latest_images(x=3):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.jpg")),
                         key=os.path.getmtime, reverse=True)[:x]

    if not image_paths:
        print("[WARN] 找不到圖片")
        return

    print(f"開始處理最近{x}張圖片")
    people_counts = []

    for path in image_paths:
        print(f"偵測: {os.path.basename(path)}")
        count = detect_image(path)
        save_result(os.path.basename(path), count)
        people_counts.append(count)

    # 計算平均人流
    avg_people = sum(people_counts) / len(people_counts)
    if avg_people == 0:
        level = "low"
        extend = 0
    elif avg_people <= 15:
        level = "medium"
        extend = 20
    else:
        level = "high"
        extend = 40

    print(f"平均人數: {avg_people:.2f}, 等級: {level}, 綠燈: {extend}秒")

    with open(CONTROL_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "avg_people": avg_people,
            "level": level,
            "extend_seconds": extend
        }, ensure_ascii=False, indent=2))


# ==============================
# 主程式
# ==============================
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
