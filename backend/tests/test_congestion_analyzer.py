
from app.services.congestion_analyzer import CongestionAnalyzer


def main():
    analyzer = CongestionAnalyzer()

    # Simulated road region: x1, y1, x2, y2
    roi = (0, 0, 1000, 1000)

    # Test 1: Low congestion
    low_vehicles = [
        {"bbox": [100, 100, 200, 200]},
    ]

    # Test 2: Moderate congestion
    moderate_vehicles = [
        {"bbox": [100, 100, 300, 300]},
        {"bbox": [350, 100, 550, 300]},
        {"bbox": [600, 100, 800, 300]},
    ]

    # Test 3: High congestion
    high_vehicles = [
        {"bbox": [0, 0, 500, 500]},
        {"bbox": [500, 0, 1000, 500]},
        {"bbox": [0, 500, 500, 1000]},
    ]

    test_cases = [
        ("Low traffic", low_vehicles, "LOW"),
        ("Moderate traffic", moderate_vehicles, "MODERATE"),
        ("High traffic", high_vehicles, "HIGH"),
    ]

    for name, vehicles, expected in test_cases:
        result = analyzer.analyze(vehicles, roi)

        print(f"\nTest: {name}")
        print(f"Vehicle count: {result['vehicle_count']}")
        print(f"Density: {result['density_percent']}%")
        print(f"Condition: {result['condition']}")
        print(f"Expected: {expected}")

        assert result["condition"] == expected, (
            f"Expected {expected}, got {result['condition']}"
        )

    print("\nAll congestion analyzer tests passed!")


if __name__ == "__main__":
    main()
