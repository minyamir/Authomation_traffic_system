
from app.services.traffic_light_recommender import (
    TrafficLightRecommender,
)


def main() -> None:
    recommender = TrafficLightRecommender()

    intersection_data = {
        "roads": {
            "north": {
                "vehicles": 8,
                "density_percent": 32.0,
                "demand": "HIGH",
            },
            "east": {
                "vehicles": 5,
                "density_percent": 18.0,
                "demand": "MODERATE",
            },
            "south": {
                "vehicles": 1,
                "density_percent": 4.0,
                "demand": "LOW",
            },
            "west": {
                "vehicles": 3,
                "density_percent": 12.0,
                "demand": "MODERATE",
            },
        }
    }

    result = recommender.recommend(intersection_data)

    print("Recommended road:", result["recommended_road"])
    print("Demand:", result["demand"])
    print("Vehicles:", result["vehicle_count"])
    print("Green duration:", result["green_seconds"], "seconds")
    print("Yellow duration:", result["yellow_seconds"], "seconds")
    print("All-red clearance:", result["all_red_seconds"], "seconds")
    print("Other roads:", result["other_roads_signal"])
    print("Road ranking:", " > ".join(
        item["road"] for item in result["ranked_roads"]
    ))

    assert result["recommended_road"] == "north"
    assert result["green_seconds"] == 30
    assert result["yellow_seconds"] == 3
    assert result["all_red_seconds"] == 2

    print("\nAll traffic-light recommender tests passed!")


if __name__ == "__main__":
    main()
