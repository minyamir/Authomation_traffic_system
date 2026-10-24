
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.services.plate_detector import PlateDetector
from app.services.plate_recognizer import PlateRecognizer


class ViolationService:
    """Coordinate plate recognition and basic speeding records."""

    def __init__(
        self,
        speed_limit_kmh: float = 50.0,
        plate_detector: PlateDetector | None = None,
        plate_recognizer: PlateRecognizer | None = None,
        min_ocr_confidence: float = 0.30,
    ) -> None:
        if speed_limit_kmh <= 0:
            raise ValueError("Speed limit must be greater than zero.")

        self.speed_limit_kmh = float(speed_limit_kmh)
        self.plate_detector = plate_detector or PlateDetector()
        self.plate_recognizer = plate_recognizer or PlateRecognizer()
        self.min_ocr_confidence = min_ocr_confidence

        self.plate_events: list[dict] = []
        self.violations: list[dict] = []
        self._seen_plate_events: set[tuple[str, str]] = set()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def process_frame(self, frame: np.ndarray) -> dict:
        """Detect plate candidates and recognize text in one frame."""
        if frame is None or frame.size == 0:
            raise ValueError("Cannot process an empty video frame.")

        recognized_plates = []

        for candidate in self.plate_detector.detect_plates(frame):
            result = self.plate_recognizer.recognize(candidate["crop"])
            plate_text = result["text"]
            confidence = result["confidence"]

            item = {
                "plate_text": plate_text,
                "ocr_confidence": confidence,
                "detection_confidence": candidate["confidence"],
                "bbox": candidate["bbox"],
                "timestamp": self._timestamp(),
            }

            recognized_plates.append(item)

            # Only save non-empty, sufficiently confident OCR results.
            if (
                plate_text
                and confidence >= self.min_ocr_confidence
            ):
                event_key = (plate_text, str(item["timestamp"])[:16])

                if event_key not in self._seen_plate_events:
                    self._seen_plate_events.add(event_key)
                    self.plate_events.append(item.copy())

        return {
            "plates": recognized_plates,
            "plate_count": len(recognized_plates),
        }

    def record_speeding(
        self,
        track_id: int | str,
        speed_kmh: float,
        plate_text: str | None = None,
    ) -> dict | None:
        """
        Record a speeding event from an existing speed estimate.

        Pass plate_text only when your application has reliably
        associated that plate with this vehicle track.
        """
        speed = float(speed_kmh)

        if speed < 0:
            raise ValueError("Vehicle speed cannot be negative.")

        if speed <= self.speed_limit_kmh:
            return None

        violation = {
            "violation_type": "speeding",
            "track_id": str(track_id),
            "plate_text": plate_text,
            "speed_kmh": round(speed, 2),
            "speed_limit_kmh": self.speed_limit_kmh,
            "excess_kmh": round(speed - self.speed_limit_kmh, 2),
            "timestamp": self._timestamp(),
        }

        self.violations.append(violation)
        return violation

    def get_summary(self) -> dict:
        """Return counts and copies of stored records."""
        return {
            "speed_limit_kmh": self.speed_limit_kmh,
            "recognized_plate_events": len(self.plate_events),
            "speeding_violations": len(self.violations),
            "plate_events": [event.copy() for event in self.plate_events],
            "violations": [item.copy() for item in self.violations],
        }

    def reset(self) -> None:
        """Clear stored results between test runs."""
        self.plate_events.clear()
        self.violations.clear()
        self._seen_plate_events.clear()
