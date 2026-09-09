import cv2
from flask import Flask, render_template, Response, request, jsonify
import threading
import imutils
import time
import numpy as np
import os
import csv
import subprocess
from datetime import datetime

import psutil

app = Flask(__name__)

# Глобальні змінні
static_back = None
frame_to_show = np.zeros((480, 640, 3), dtype=np.uint8)
lock = threading.Lock()

# Змінні для відстеження руху та параметрів
motion_detected = False
last_motion_time = "Руху ще не було"
min_contour_area = 10000

# Змінні для збереження стану сесії руху
is_moving = False
motion_start_time = None

# Створення папки для знімків
os.makedirs('snapshots', exist_ok=True)

# Створення CSV-файлу з заголовками, якщо він не існує
if not os.path.exists('motion_log.csv'):
    with open('motion_log.csv', mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Start Time', 'End Time'])

def log_motion(start_time, end_time):
    """Запис початку та завершення руху в CSV-файл"""
    with open('motion_log.csv', mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S")
        ])

def capture_and_detect_motion():
    global static_back, frame_to_show, motion_detected, last_motion_time
    global is_moving, motion_start_time, min_contour_area

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        check, frame = video.read()
        if not check:
            continue
            
        frame = imutils.resize(frame, width=640)
        motion = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if static_back is None:
            static_back = gray.copy().astype("float")
            continue
            
        cv2.accumulateWeighted(gray, static_back, 0.05)
        diff_frame = cv2.absdiff(gray, cv2.convertScaleAbs(static_back))
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with lock:
            current_min_area = min_contour_area

        for contour in cnts:
            if cv2.contourArea(contour) < current_min_area:
                continue
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Логіка фіксації руху, збереження знімків та CSV-журналювання
        if motion == 1:
            if not is_moving:
                is_moving = True
                motion_start_time = now
                # Збереження знімка при початку руху
                filename = f"snapshots/snapshot_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
            
            last_motion_time = timestamp_str
        else:
            if is_moving:
                is_moving = False
                log_motion(motion_start_time, now)

        cv2.putText(frame, f"Motion: {'YES' if motion else 'NO'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if motion else (0, 255, 0), 2)
        cv2.putText(frame, timestamp_str, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        with lock:
            frame_to_show = frame.copy()
            motion_detected = bool(motion)
            
        time.sleep(0.03)

def generate():
    global frame_to_show
    while True:
        with lock:
            frame = frame_to_show.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

def get_gpu_usage():
    """Повертає завантаження NVIDIA GPU або None, якщо GPU недоступний."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError, subprocess.TimeoutExpired):
        pass
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Ендпоінт для отримання часу останнього руху"""
    with lock:
        return jsonify({
            'last_motion': last_motion_time,
            'is_moving': motion_detected
        })

@app.route('/metrics')
def metrics():
    """Ендпоінт для поточних показників CPU, GPU та оперативної пам'яті."""
    memory = psutil.virtual_memory()
    return jsonify({
        'cpu': psutil.cpu_percent(interval=0.1),
        'memory': memory.percent,
        'gpu': get_gpu_usage()
    })

@app.route('/set_sensitivity', methods=['POST'])
def set_sensitivity():
    """Ендпоінт для зміни порогової площі контуру"""
    global min_contour_area
    data = request.get_json()
    if data and 'sensitivity' in data:
        with lock:
            min_contour_area = int(data['sensitivity'])
        return jsonify({'status': 'success', 'min_contour_area': min_contour_area})
    return jsonify({'status': 'error'}), 400

if __name__ == '__main__':
    threading.Thread(target=capture_and_detect_motion, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)