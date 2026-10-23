
from app.services.intersection_analyzer import IntersectionAnalyzer


def main():
    analyzer = IntersectionAnalyzer()

    # Simulated four-way intersection.
    # ROI format: (x1, y1, x2, y2)
    road_regions = {
        "north": (400, 0, 600, 400),
        "east": (600, 400, 1000, 600),
        "south": (400, 600, 600, 1000),
        "west": (0, 400, 400, 600),
    }

    # Simulated vehicle bounding boxes.
    # North: 3 vehicles, East: 2, South: 1, West: 2.
    vehicles = [
        # North
        {"track_id": 1, "class_name": "car", "bbox": [420, 40, 480, 100]},
        {"track_id": 2, "class_name": "car", "bbox": [420, 150, 480, 210]},
        {"track_id": 3, "class_name": "truck", "bbox": [420, 260, 490, 330]},

        # East
        {"track_id": 4, "class_name": "car", "bbox": [650, 420, 710, 480]},
        {"track_id": 5, "class_name": "bus", "bbox": [800, 420, 870, 490]},

        # South
        {"track_id": 6, "class_name": "motorcycle", "bbox": [420, 800, 460, 840]},

        # West
        {"track_id": 7, "class_name": "car", "bbox": [50, 420, 110, 480]},
        {"track_id": 8, "class_name": "car", "bbox": [220, 420, 280, 480]},
    ]

    result = analyzer.analyze(vehicles, road_regions)

    print("\n========== INTERSECTION ANALYSIS ==========")

    for approach, data in result["roads"].items():
        print(
            f"{approach.upper():<7} | "
            f"Vehicles: {data['vehicles']} | "
            f"Density: {data['density_percent']:.2f}% | "
            f"Demand: {data['demand']}"
        )

    print("\nRoad ranking:", " > ".join(
        road.upper() for road in result["ranked_approaches"]
    ))
    print("Highest-demand road:", result["highest_demand"].upper())

    # Verify the analyzer results.
    assert result["roads"]["north"]["vehicles"] == 3
    assert result["roads"]["east"]["vehicles"] == 2
    assert result["roads"]["south"]["vehicles"] == 1
    assert result["roads"]["west"]["vehicles"] == 2

    assert result["highest_demand"] == "north"

    print("\nAll intersection analyzer tests passed!")


if __name__ == "__main__":
    main()
