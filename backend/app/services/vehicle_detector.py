
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """Detect cars, motorcycles, buses, and trucks in a video frame."""

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        image_size: int = 960,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self._model: YOLO | None = None

    def _load_model(self) -> YOLO:
        if self._model is None:
            model_file = Path(self.model_path)

            if model_file.is_file():
                self._model = YOLO(str(model_file))
            else:
                self._model = YOLO(self.model_path)

        return self._model

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return vehicle classes, confidence, and original-frame boxes."""

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array.")

        if frame.size == 0:
            raise ValueError("frame cannot be empty.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image.")

        results = self._load_model().predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=0.45,
            classes=list(self.VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": self.VEHICLE_CLASSES[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                })

        return detections
