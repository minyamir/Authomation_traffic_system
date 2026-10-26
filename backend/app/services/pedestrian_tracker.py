from typing import Any

import numpy as np

from app.services.pedestrian_detector import PedestrianDetector


class PedestrianTracker:
    """Assign and maintain pedestrian IDs across video frames."""

    def __init__(
        self,
        detector: PedestrianDetector | None = None,
        max_distance: float = 80.0,
        max_missed_frames: int = 10,
    ) -> None:
        self.detector = detector or PedestrianDetector()
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        self.next_track_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, width, height = bbox
        return x + width / 2, y + height / 2

    def update(
        self,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Detect people and match them to existing track IDs."""
        detections = self.detector.detect(frame)

        # Age all existing tracks; matched tracks reset this counter.
        for track in self.tracks.values():
            track["missed"] += 1

        unmatched_ids = set(self.tracks.keys())
        tracked_people = []

        # Match larger detections first.
        detections.sort(
            key=lambda person: (
                person["bbox"][2] * person["bbox"][3]
            ),
            reverse=True,
        )

        for person in detections:
            center = self._center(person["bbox"])
            best_id = None
            best_distance = self.max_distance

            for track_id in unmatched_ids:
                old_center = self.tracks[track_id]["center"]

                distance = float(
                    np.hypot(
                        center[0] - old_center[0],
                        center[1] - old_center[1],
                    )
                )

                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[best_id] = {
                    "center": center,
                    "bbox": person["bbox"],
                    "missed": 0,
                }
            else:
                unmatched_ids.remove(best_id)

                self.tracks[best_id]["center"] = center
                self.tracks[best_id]["bbox"] = person["bbox"]
                self.tracks[best_id]["missed"] = 0

            tracked_people.append({
                **person,
                "track_id": best_id,
                "center": center,
            })

        # Remove tracks not seen for too long.
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed_frames
        ]

        for track_id in expired_ids:
            del self.tracks[track_id]

        return tracked_people

    def track(
        self,
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Alias for update(), useful in video-processing code."""
        return self.update(frame)