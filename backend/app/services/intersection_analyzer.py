
from typing import Any

from app.services.congestion_analyzer import CongestionAnalyzer


class IntersectionAnalyzer:
    """Analyze demand independently for each approach of a crossroads."""

    APPROACHES = ("north", "east", "south", "west")

    def __init__(
        self,
        low_threshold: float = 10.0,
        high_threshold: float = 25.0,
    ) -> None:
        self.congestion_analyzer = CongestionAnalyzer(
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        self.last_selected: str | None = None

    def analyze(
        self,
        vehicles: list[dict[str, Any]],
        road_regions: dict[str, tuple[int, int, int, int]],
    ) -> dict[str, Any]:
        """
        Analyze vehicle demand in four road regions.

        Args:
            vehicles: Vehicle records from VehicleTracker.track(frame).
            road_regions: Mapping of approach names to (x1, y1, x2, y2).

        Returns:
            Demand and congestion data for each approach, plus the
            recommended highest-demand approach.

        Note:
            Each road region must cover its approach, not the whole
            intersection or the entire video frame.
        """

        missing = set(self.APPROACHES) - set(road_regions)
        if missing:
            raise ValueError(
                f"Missing road regions: {', '.join(sorted(missing))}"
            )

        road_results: dict[str, dict[str, Any]] = {}

        for approach in self.APPROACHES:
            roi = road_regions[approach]

            congestion = self.congestion_analyzer.analyze(
                vehicles,
                roi,
            )

            vehicle_count = congestion["vehicle_count"]
            density = congestion["density_percent"]

            road_results[approach] = {
                "vehicles": vehicle_count,
                "density_percent": density,
                "demand": congestion["condition"],
            }

        # Rank demand by density first, then by vehicle count.
        # Rotate between tied roads to reduce repeated favoritism.
        highest_density = max(
            result["density_percent"]
            for result in road_results.values()
        )

        candidates = [
            approach
            for approach, result in road_results.items()
            if result["density_percent"] == highest_density
        ]

        if len(candidates) > 1:
            highest_count = max(
                road_results[name]["vehicles"]
                for name in candidates
            )
            candidates = [
                name
                for name in candidates
                if road_results[name]["vehicles"] == highest_count
            ]

        if self.last_selected in candidates:
            previous_index = candidates.index(self.last_selected)
            selected = candidates[
                (previous_index + 1) % len(candidates)
            ]
        else:
            selected = candidates[0]

        self.last_selected = selected

        # Rank all approaches for display.
        ranked_approaches = sorted(
            self.APPROACHES,
            key=lambda name: (
                road_results[name]["density_percent"],
                road_results[name]["vehicles"],
            ),
            reverse=True,
        )

        return {
            "roads": road_results,
            "highest_demand": selected,
            "ranked_approaches": ranked_approaches,
        }
