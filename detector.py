from ultralytics import YOLO
import cv2
import tempfile
import os

# ---------------- Load YOLO Model ---------------- #

model = YOLO("yolov8n.pt")
def get_model():
    return model


# ---------------- Image Detection ---------------- #

def detect_image(image, confidence):

    results = model.predict(
        image,
        conf=confidence,
        verbose=False
    )

    annotated = results[0].plot()

    names = model.names

    counts = {}

    for box in results[0].boxes:

        cls = int(box.cls[0])

        name = names[cls]

        counts[name] = counts.get(name, 0) + 1

    return annotated, counts


# ---------------- Video Detection + Tracking ---------------- #

def detect_video(video_file, confidence):

    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_input.write(video_file.read())

    temp_input.close()

    output_path = os.path.join(
        "output",
        "detected_video.mp4"
    )

    cap = cv2.VideoCapture(temp_input.name)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    object_counter = {}

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            conf=confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated = results[0].plot()

        out.write(annotated)

        if results[0].boxes is not None:

            for box in results[0].boxes:

                cls = int(box.cls[0])

                name = model.names[cls]

                object_counter[name] = (
                    object_counter.get(name, 0) + 1
                )

    cap.release()

    out.release()

    os.remove(temp_input.name)

    return output_path, object_counter