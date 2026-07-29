# Aerial Crime (Violence) Detection

End-to-end project: train a video-classification model on your `Violence_Dataset`
in Google Colab, then run it live through a Flask GUI that ingests drone footage,
draws a border/label when violence is detected, and records the annotated output.

## Project layout

```
aerial_crime_detection/
├── train_and_evaluate.ipynb      # Colab notebook: train + Accuracy/Precision/Recall/F1
└── flask_app/
    ├── app.py                    # Flask server (upload + live/RTSP inference)
    ├── requirements.txt
    ├── model/                    # put violence_detection_model.h5 here
    ├── templates/
    │   ├── index.html
    │   ├── result.html
    │   └── live.html
    ├── uploads/                  # user-uploaded videos land here
    └── outputs/                  # annotated output videos land here
```

## Step 1 — Train the model on Colab

1. Zip your `Violence_Dataset` folder (with `train/val/test` → `Violence/NonViolence`)
   and upload it to your Google Drive, then unzip it there.
2. Open `train_and_evaluate.ipynb` in Google Colab (Runtime → Change runtime type → GPU).
3. Update the `DATASET_DIR` variable in the "Mount Drive" cell to match your Drive path.
4. Run all cells top to bottom. The notebook will:
   - Extract 16 evenly-sampled frames per clip, resized to 64×64.
   - Train a **MobileNetV2 (per-frame features) + Bidirectional LSTM** classifier.
   - Print **Accuracy, Precision, Recall, F1-score** and a confusion matrix on the test set.
   - Save the trained model to `violence_detection_model.h5` in your Drive.
5. Download `violence_detection_model.h5` and place it at:
   `flask_app/model/violence_detection_model.h5`

Notes:
- 64×64 frames / 16-frame sequences keep this trainable on Colab's free tier. If you have
  a lot of data and a Pro GPU, you can raise `IMAGE_HEIGHT`/`IMAGE_WIDTH` (e.g. to 96 or 128)
  and `SEQUENCE_LENGTH` for better accuracy at the cost of speed — just mirror the change in
  `app.py` (`IMAGE_HEIGHT`, `IMAGE_WIDTH`, `SEQUENCE_LENGTH` must match between training and inference).
- If a class folder has far more videos than the other, consider `class_weight` in `model.fit`
  to handle imbalance — ask me and I'll add it.

## Step 2 — Run the Flask app

```bash
cd flask_app
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** in your browser.

### Option A — Analyze a recorded drone video
Upload a video file. The app runs a sliding 16-frame window through the model,
overlays a **red border + "Violence"** label (or **green + "NonViolence"**) on every
frame, and shows you the annotated result plus a violence-frame ratio and verdict.
The output is saved under `flask_app/outputs/annotated_<name>.mp4`.

### Option B — Live / drone stream
On the home page, under "Live / drone stream":
- Leave source as `0` to use your machine's webcam, **or**
- Enter your drone's stream URL, e.g. `rtsp://192.168.1.50:8554/live` (most drones/gimbal
  companion computers expose RTSP or an HTTP MJPEG endpoint — check your drone SDK's docs).
- Check "Record" to also save the annotated live feed to `outputs/live_recording.mp4`.

This opens an MJPEG stream (`/live_feed`) that's annotated in real time frame-by-frame.

## How detection works (summary)

1. Frames are read from the video/stream and resized to 64×64, normalized to [0,1].
2. The last 16 frames form a rolling window fed to the trained model.
3. The model outputs `[P(NonViolence), P(Violence)]`; the higher-probability class
   becomes the current label, shown until the next window prediction updates it.
4. `draw_overlay()` draws a thick red rectangle border + label text when the current
   label is "Violence", and a thin green border otherwise — this is what gets
   burned into the recorded output video.

## Extending this project

- **Real drone integration**: most drones (DJI, ArduPilot/PX4 companion computers) can
  push an RTSP/RTMP stream from the gimbal camera — point `source` at that URL.
- **Alerting**: hook a webhook/SMS/email call into `app.py` wherever `state['label'] ==
  'Violence'` is set, to notify an operator in real time.
- **Better temporal modeling**: swap the MobileNetV2+BiLSTM head for a 3D CNN
  (e.g. `torchvision.models.video.r3d_18`) if you want to explore PyTorch instead.
- **Geolocation overlay**: if your drone telemetry (GPS) is available, overlay
  coordinates on frames where violence is flagged for easier dispatch.
# aerial_crime_detection
