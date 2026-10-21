
from typing import Any

import numpy as np

from app.services.vehicle_detector import VehicleDetector


class VehicleTracker:
    """Track vehicles across consecutive video frames."""

    def __init__(self) -> None:
        self.detector = VehicleDetector()

    def track(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect vehicles and assign persistent tracking IDs."""

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array.")

        if frame.size == 0:
            raise ValueError("frame cannot be empty.")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame must be a three-channel BGR image."
            )

        model = self.detector._load_model()

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.detector.image_size,
            conf=self.detector.confidence_threshold,
            classes=list(self.detector.VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        vehicles: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            track_ids = boxes.id

            for index, box in enumerate(boxes):
                class_id = int(box.cls[0].item())

                class_name = self.detector.VEHICLE_CLASSES.get(
                    class_id
                )

                if class_name is None:
                    continue

                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().tolist()
                )

                track_id = None

                if track_ids is not None:
                    track_id = int(track_ids[index].item())

                vehicles.append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                })

        return vehicles
