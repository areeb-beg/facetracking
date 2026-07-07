import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
MODEL_FILE = "blaze_face_short_range.tflite"


def _ensure_model() -> str:
    import os
    import urllib.request

    if not os.path.isfile(MODEL_FILE):
        print(f"Downloading face detection model ({MODEL_URL})...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Download complete.")
    return MODEL_FILE


def _draw_detections(frame: np.ndarray, detection_result) -> None:
    h, w, _ = frame.shape

    for detection in detection_result.detections:
        box = detection.bounding_box
        x = box.origin_x
        y = box.origin_y
        bw = box.width
        bh = box.height

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        score = detection.categories[0].score
        label = f"{score:.2%}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main() -> None:
    model_path = _ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.5,
    )

    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            result = detector.detect(mp_image)

            _draw_detections(frame, result)

            cv2.imshow("School Reception Robot - Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
