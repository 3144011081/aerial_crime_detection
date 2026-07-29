"""
Aerial Crime (Violence) Detection - Flask GUI
------------------------------------------------
Feed a drone video (file upload) or a live/RTSP stream. The trained
MobileNetV2 + BiLSTM model classifies a rolling window of frames as
Violence / NonViolence. When Violence is detected, a red border is
drawn around the frame and the label is overlaid. The annotated
output is recorded to disk as an .mp4 file.
"""

import os
import cv2
import numpy as np
from flask import (Flask, render_template, request, Response,
                    redirect, url_for, send_from_directory, flash)
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'violence_detection_model.h5')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ['NonViolence', 'Violence']   # must match training notebook order

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
app.secret_key = 'change-this-secret-key'

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
model = None


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f'Model file not found at {MODEL_PATH}. '
                'Train it in the Colab notebook and place violence_detection_model.h5 there.'
            )
        print('Loading model...')
        model = load_model(MODEL_PATH)
        print('Model loaded.')
    return model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_overlay(frame, label, confidence):
    """Draws a colored border + label text on the frame based on the prediction."""
    h, w = frame.shape[:2]
    is_violence = (label == 'Violence')
    color = (0, 0, 255) if is_violence else (0, 200, 0)      # BGR: red / green
    thickness = 14 if is_violence else 4

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

    text = f'{label} ({confidence * 100:.1f}%)'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 15), color, -1)
    cv2.putText(frame, text, (15, 20 + th), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    return frame


# --------------------------------------------------------------------------
# Core inference loop shared by "process a file" and "live stream" paths
# --------------------------------------------------------------------------
def run_inference_on_frame(frame, frame_buffer, state):
    """Updates frame_buffer + state in place, returns (label, confidence)."""
    resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0
    frame_buffer.append(normalized)

    if len(frame_buffer) >= SEQUENCE_LENGTH:
        window = np.expand_dims(np.array(frame_buffer[-SEQUENCE_LENGTH:]), axis=0)
        preds = get_model().predict(window, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        state['label'] = CLASSES_LIST[class_idx]
        state['confidence'] = float(preds[class_idx])
        # slide the window forward by one frame instead of recomputing from scratch
        frame_buffer.pop(0)

    return state['label'], state['confidence']


def process_video_file(input_path, output_path):
    """Reads a video file, annotates every frame, writes an annotated .mp4."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {input_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_buffer = []
    state = {'label': 'NonViolence', 'confidence': 0.0}
    violence_frame_count = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        label, confidence = run_inference_on_frame(frame, frame_buffer, state)
        if label == 'Violence':
            violence_frame_count += 1

        frame = draw_overlay(frame, label, confidence)
        writer.write(frame)

    cap.release()
    writer.release()

    violence_ratio = violence_frame_count / total_frames if total_frames else 0.0
    return {
        'total_frames': total_frames,
        'violence_frames': violence_frame_count,
        'violence_ratio': violence_ratio,
        'verdict': 'VIOLENCE DETECTED' if violence_ratio > 0.15 else 'No significant violence detected',
    }


def generate_live_frames(source, record_path=None):
    """Generator for MJPEG streaming; optionally records the annotated feed to disk."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return

    writer = None
    if record_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(record_path, fourcc, fps, (width, height))

    frame_buffer = []
    state = {'label': 'NonViolence', 'confidence': 0.0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            label, confidence = run_inference_on_frame(frame, frame_buffer, state)
            frame = draw_overlay(frame, label, confidence)

            if writer is not None:
                writer.write(frame)

            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        if writer is not None:
            writer.release()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files or request.files['video'].filename == '':
        flash('Please choose a video file to upload.')
        return redirect(url_for('index'))

    file = request.files['video']
    if not allowed_file(file.filename):
        flash('Unsupported file type. Use mp4, avi, mov, or mkv.')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    base_name = os.path.splitext(filename)[0]
    output_filename = f'annotated_{base_name}.mp4'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        stats = process_video_file(input_path, output_path)
    except Exception as e:
        flash(f'Processing failed: {e}')
        return redirect(url_for('index'))

    return render_template('result.html', video_file=output_filename, stats=stats)


@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/live_view')
def live_view():
    # source: '0' for local webcam, or an rtsp://... URL for a drone stream
    source = request.args.get('source', '0')
    record = request.args.get('record', '0') == '1'
    return render_template('live.html', source=source, record=record)


@app.route('/live_feed')
def live_feed():
    source = request.args.get('source', '0')
    record = request.args.get('record', '0') == '1'
    try:
        cam_source = int(source)
    except ValueError:
        cam_source = source  # RTSP/HTTP stream URL from the drone

    record_path = None
    if record:
        record_path = os.path.join(OUTPUT_FOLDER, 'live_recording.mp4')

    return Response(generate_live_frames(cam_source, record_path),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    get_model()  # fail fast on startup if the model file is missing
    app.run(debug=True, host='0.0.0.0', port=5000)
