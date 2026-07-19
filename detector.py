from ultralytics import YOLO
import cv2
import tempfile
from collections import Counter

# ---------------- MODEL ---------------- #

MODEL = YOLO("yolov8s.pt")   # Better than yolov8n


def get_model():
    return MODEL


# ---------------- IMAGE DETECTION ---------------- #

def detect_image(image, confidence=0.4):

    results = MODEL.predict(
        image,
        conf=confidence,
        verbose=False
    )

    annotated = results[0].plot()

    counts = Counter()

    names = MODEL.names

    if len(results[0].boxes):

        for cls in results[0].boxes.cls:

            counts[names[int(cls)]] += 1

    return annotated, dict(counts)


# ---------------- VIDEO DETECTION ---------------- #

def detect_video(uploaded_video, confidence=0.4):

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    temp_input.write(uploaded_video.read())

    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    writer = cv2.VideoWriter(

        output_path,

        cv2.VideoWriter_fourcc(*"mp4v"),

        fps,

        (width, height)

    )

    counts = Counter()

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = MODEL.track(

            frame,

            conf=confidence,

            persist=True,

            tracker="bytetrack.yaml",

            verbose=False

        )

        annotated = results[0].plot()

        writer.write(annotated)

        names = MODEL.names

        if len(results[0].boxes):

            for cls in results[0].boxes.cls:

                counts[names[int(cls)]] += 1

    cap.release()

    writer.release()

    return output_path, dict(counts)