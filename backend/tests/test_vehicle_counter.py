
from app.services.vehicle_counter import VehicleCounter


def main():
    counter = VehicleCounter()

    frame_1 = [
        {"track_id": 1, "class_name": "car"},
        {"track_id": 2, "class_name": "motorcycle"},
    ]

    frame_2 = [
        {"track_id": 1, "class_name": "car"},
        {"track_id": 2, "class_name": "motorcycle"},
        {"track_id": 3, "class_name": "bus"},
    ]

    print("Frame 1:", counter.update(frame_1))
    print("Frame 2:", counter.update(frame_2))


if __name__ == "__main__":
    main()
