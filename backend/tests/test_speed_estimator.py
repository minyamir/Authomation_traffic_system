
from app.services.speed_estimator import SpeedEstimator


def main() -> None:
    estimator = SpeedEstimator(
        pixels_per_meter=10.0,
        speed_limit_kmh=50.0,
        min_elapsed_seconds=0.1,
    )

    # Frame 1: first observation, so speed is not available yet.
    vehicles = [
        {
            "track_id": 1,
            "class_name": "car",
            "bbox": [90, 90, 110, 110],
        }
    ]

    result = estimator.estimate(vehicles, timestamp=1.0)
    assert result[0]["speed_kmh"] is None
    print("Frame 1:", result)

    # Frame 2: moves 10 pixels in one second.
    # At 10 pixels per meter, that equals 1 meter/second = 3.6 km/h.
    vehicles = [
        {
            "track_id": 1,
            "class_name": "car",
            "bbox": [100, 90, 120, 110],
        }
    ]

    result = estimator.estimate(vehicles, timestamp=2.0)

    assert result[0]["speed_kmh"] == 3.6
    assert result[0]["speeding"] is False

    print("Frame 2:", result)
    print("PASS: speed estimation works.")


if __name__ == "__main__":
    main()
