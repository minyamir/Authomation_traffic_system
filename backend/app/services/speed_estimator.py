
from collections import defaultdict, deque
from typing import Any

import math
import time

import numpy as np


class SpeedEstimator:
    """Estimate vehicle speed using tracked positions and road calibration."""

    def __init__(
        self,
        pixels_per_meter: float = 10.0,
        speed_limit_kmh: float = 50.0,
        history_size: int = 5,
        min_elapsed_seconds: float = 0.1,
    ) -> None:
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be greater than zero.")

        if speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh must be greater than zero.")

        if history_size < 1:
            raise ValueError("history_size must be at least 1.")

        self.pixels_per_meter = pixels_per_meter
        self.speed_limit_kmh = speed_limit_kmh
        self.min_elapsed_seconds = min_elapsed_seconds

        self.previous_positions: dict[int, tuple[float, float]] = {}
        self.previous_times: dict[int, float] = {}

        self.speed_history: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    @staticmethod
    def _vehicle_center(
        vehicle: dict[str, Any],
    ) -> tuple[float, float]:
        """Return the center point of a vehicle bounding box."""

        bbox = vehicle.get("bbox")

        if bbox is None or len(bbox) != 4:
            raise ValueError("Each vehicle must contain bbox=[x1, y1, x2, y2].")

        x1, y1, x2, y2 = map(float, bbox)

        return (x1 + x2) / 2, (y1 + y2) / 2

    def estimate(
        self,
        vehicles: list[dict[str, Any]],
        timestamp: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Estimate speed for tracked vehicles.

        Each vehicle must contain:
            track_id: persistent tracking ID
            bbox: [x1, y1, x2, y2]

        timestamp is measured in seconds. If omitted, monotonic time is used.
        The first observation of a vehicle has no speed estimate.
        """

        current_time = time.monotonic() if timestamp is None else float(timestamp)

        if not math.isfinite(current_time):
            raise ValueError("timestamp must be finite.")

        results: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        for vehicle in vehicles:
            if "track_id" not in vehicle or vehicle["track_id"] is None:
                continue

            track_id = int(vehicle["track_id"])

            if track_id < 0:
                continue

            if track_id in seen_ids:
                continue

            seen_ids.add(track_id)

            center_x, center_y = self._vehicle_center(vehicle)

            previous = self.previous_positions.get(track_id)
            previous_time = self.previous_times.get(track_id)

            speed_kmh: float | None = None

            if previous is not None and previous_time is not None:
                elapsed = current_time - previous_time

                if elapsed >= self.min_elapsed_seconds:
                    pixel_distance = math.hypot(
                        center_x - previous[0],
                        center_y - previous[1],
                    )

                    distance_meters = (
                        pixel_distance / self.pixels_per_meter
                    )

                    raw_speed_kmh = (
                        distance_meters / elapsed
                    ) * 3.6

                    self.speed_history[track_id].append(raw_speed_kmh)

                    speed_kmh = sum(
                        self.speed_history[track_id]
                    ) / len(self.speed_history[track_id])

                    self.previous_positions[track_id] = (
                        center_x,
                        center_y,
                    )
                    self.previous_times[track_id] = current_time

                else:
                    # Ignore duplicate or too-close timestamps.
                    speed_kmh = None
            else:
                self.previous_positions[track_id] = (
                    center_x,
                    center_y,
                )
                self.previous_times[track_id] = current_time

            result = vehicle.copy()
            result["speed_kmh"] = (
                round(speed_kmh, 2)
                if speed_kmh is not None
                else None
            )
            result["speeding"] = (
                speed_kmh is not None
                and speed_kmh > self.speed_limit_kmh
            )
            result["speed_limit_kmh"] = self.speed_limit_kmh

            results.append(result)

        return results

    def reset(self) -> None:
        """Clear stored positions and speed history."""

        self.previous_positions.clear()
        self.previous_times.clear()
        self.speed_history.clear()
