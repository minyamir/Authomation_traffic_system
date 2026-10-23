
from typing import Any


class TrafficLightRecommender:
    """Recommend a traffic-light phase from intersection demand."""

    def __init__(
        self,
        min_green: int = 10,
        max_green: int = 30,
        yellow_seconds: int = 3,
        all_red_seconds: int = 2,
    ) -> None:
        if min_green <= 0:
            raise ValueError("min_green must be positive.")

        if max_green < min_green:
            raise ValueError("max_green must be >= min_green.")

        if yellow_seconds <= 0 or all_red_seconds <= 0:
            raise ValueError("Clearance durations must be positive.")

        self.min_green = min_green
        self.max_green = max_green
        self.yellow_seconds = yellow_seconds
        self.all_red_seconds = all_red_seconds

    def recommend(
        self,
        intersection_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Recommend the next road and its green duration."""

        roads = intersection_data.get("roads", {})

        if not roads:
            raise ValueError("Intersection data must contain roads.")

        valid_roads = {}

        for road, data in roads.items():
            if not isinstance(data, dict):
                continue

            try:
                count = int(data["vehicles"])
                density = float(data["density_percent"])
                demand = str(data["demand"]).upper()
            except (KeyError, TypeError, ValueError):
                continue

            if count < 0 or density < 0 or demand not in {
                "LOW", "MODERATE", "HIGH"
            }:
                continue

            valid_roads[road] = {
                "vehicles": count,
                "density_percent": density,
                "demand": demand,
            }

        if not valid_roads:
            raise ValueError("No valid road-demand data was provided.")

        # Highest demand first; density and vehicle count break ties.
        priority = {"LOW": 1, "MODERATE": 2, "HIGH": 3}

        ranked_roads = sorted(
            valid_roads.items(),
            key=lambda item: (
                priority[item[1]["demand"]],
                item[1]["density_percent"],
                item[1]["vehicles"],
            ),
            reverse=True,
        )

        selected_road, selected_data = ranked_roads[0]
        demand = selected_data["demand"]

        # More demand earns more green time.
        green_by_demand = {
            "LOW": self.min_green,
            "MODERATE": (self.min_green + self.max_green) // 2,
            "HIGH": self.max_green,
        }

        green_seconds = green_by_demand[demand]

        return {
            "recommended_road": selected_road,
            "demand": demand,
            "vehicle_count": selected_data["vehicles"],
            "density_percent": selected_data["density_percent"],
            "green_seconds": green_seconds,
            "yellow_seconds": self.yellow_seconds,
            "all_red_seconds": self.all_red_seconds,
            "other_roads_signal": "RED",
            "phase_sequence": [
                "GREEN",
                "YELLOW",
                "ALL_RED",
                "NEXT_GREEN",
            ],
            "ranked_roads": [
                {
                    "road": road,
                    **data,
                }
                for road, data in ranked_roads
            ],
            "note": (
                "Recommendation only. This service does not control "
                "physical traffic signals."
            ),
        }
