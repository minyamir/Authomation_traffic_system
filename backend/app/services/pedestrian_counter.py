from typing import Any


class PedestrianCounter:
    """Count unique tracked pedestrians entering a crossing zone."""

    def __init__(
        self,
        crossing_zone: tuple[int, int, int, int],
    ) -> None:
        x, y, width, height = crossing_zone

        if width <= 0 or height <= 0:
            raise ValueError(
                "Crossing zone width and height must be positive"
            )

        self.crossing_zone = crossing_zone
        self.counted_ids: set[int] = set()
        self.total_crossings = 0

    def is_inside(self, point: tuple[float, float]) -> bool:
        """Return whether a point lies inside the crossing rectangle."""
        x, y, width, height = self.crossing_zone
        px, py = point

        return (
            x <= px <= x + width
            and y <= py <= y + height
        )

    def update(
        self,
        pedestrians: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Count each track ID once when it enters the zone."""
        current_inside_ids = set()
        new_crossings = 0

        for person in pedestrians:
            track_id = person.get("track_id")
            center = person.get("center")

            if track_id is None or center is None:
                continue

            if self.is_inside(center):
                current_inside_ids.add(track_id)

                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.total_crossings += 1
                    new_crossings += 1

        return {
            "total_crossings": self.total_crossings,
            "people_inside": len(current_inside_ids),
            "new_crossings": new_crossings,
            "crossing_zone": self.crossing_zone,
        }

    def draw_zone(self, frame):
        """Draw the configured crossing zone on a frame."""
        import cv2

        annotated = frame.copy()
        x, y, width, height = self.crossing_zone

        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (255, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            f"Crossings: {self.total_crossings}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def reset(self) -> None:
        """Reset counts for a new video or test."""
        self.counted_ids.clear()
        self.total_crossings = 0