
from typing import Any


class CongestionAnalyzer:
    """Estimate traffic congestion from vehicles inside a road region."""

    def __init__(
        self,
        low_threshold: float = 10.0,
        high_threshold: float = 25.0,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def analyze(
        self,
        vehicles: list[dict[str, Any]],
        roi: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        """
        Estimate image-space vehicle occupancy in a road region.

        roi format: (x1, y1, x2, y2)
        """

        x1, y1, x2, y2 = roi

        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI must have positive width and height.")

        roi_area = (x2 - x1) * (y2 - y1)

        vehicle_count = 0
        occupied_area = 0

        for vehicle in vehicles:
            bbox = vehicle.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            vx1, vy1, vx2, vy2 = bbox

            # Find the part of the vehicle box inside the road region.
            ix1 = max(x1, vx1)
            iy1 = max(y1, vy1)
            ix2 = min(x2, vx2)
            iy2 = min(y2, vy2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # Count vehicles whose box centers lie inside the ROI.
            center_x = (vx1 + vx2) / 2
            center_y = (vy1 + vy2) / 2

            if x1 <= center_x < x2 and y1 <= center_y < y2:
                vehicle_count += 1

                occupied_area += (ix2 - ix1) * (iy2 - iy1)

        # Approximate percentage of the road image covered by vehicles.
        density_percent = min(
            100.0,
            (occupied_area / roi_area) * 100.0,
        )

        if density_percent < self.low_threshold:
            condition = "LOW"
        elif density_percent < self.high_threshold:
            condition = "MODERATE"
        else:
            condition = "HIGH"

        return {
            "vehicle_count": vehicle_count,
            "density_percent": round(density_percent, 2),
            "condition": condition,
        }
