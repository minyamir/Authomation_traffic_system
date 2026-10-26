from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class PedestrianDetector:
    """Detect people in traffic video frames using YOLO."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        image_size: int = 640,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return detected people with bounding boxes and confidence."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")

        if frame.size == 0:
            raise ValueError("frame must not be empty")

        predictions = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
          )

        pedestrians = []

        for prediction in predictions:
            if prediction.boxes is None:
                continue

            for box in prediction.boxes:
                x1, y1, x2, y2 = (
                    int(value)
                    for value in box.xyxy[0].tolist()
                )

                score = float(box.conf[0].item())

                pedestrians.append({
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "confidence": score,
                    "class_id": self.PERSON_CLASS_ID,
                    "class_name": "person",
                })

        return pedestrians

    def detect_pedestrians(
        self,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Explicitly named alias for pedestrian detection."""
        return self.detect(frame)

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        pedestrians: list[dict[str, Any]],
    ) -> np.ndarray:
        """Draw detected people on a copy of the frame."""
        annotated = frame.copy()

        for person in pedestrians:
            x, y, width, height = person["bbox"]
            confidence = person["confidence"]

            cv2.rectangle(
                annotated,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            label = f"Pedestrian {confidence:.0%}"
            cv2.putText(
                annotated,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated