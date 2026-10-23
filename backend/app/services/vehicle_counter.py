
from typing import Any


class VehicleCounter:
    """Count unique tracked vehicles and classify them."""

    def __init__(self) -> None:
        self.seen_track_ids: set[int] = set()

        self.class_counts: dict[str, int] = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

    def update(
        self,
        vehicles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Count each tracked vehicle ID only once."""

        for vehicle in vehicles:
            track_id = vehicle.get("track_id")
            class_name = vehicle.get("class_name")

            # Ignore vehicles without a tracking ID.
            if track_id is None:
                continue

            # Count a vehicle only the first time its ID appears.
            if track_id not in self.seen_track_ids:
                self.seen_track_ids.add(track_id)

                if class_name in self.class_counts:
                    self.class_counts[class_name] += 1

        return {
            "total_vehicles": len(self.seen_track_ids),
            "class_counts": self.class_counts.copy(),
        }

    def reset(self) -> None:
        """Reset counts before processing a new video."""

        self.seen_track_ids.clear()

        for class_name in self.class_counts:
            self.class_counts[class_name] = 0
