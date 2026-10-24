
import cv2

from app.services.vehicle_tracker import VehicleTracker
from app.services.vehicle_counter import VehicleCounter
from app.services.speed_estimator import SpeedEstimator


def main() -> None:
    # 1. Get the video path
    video_path = input(
        "Enter the path to your traffic video (e.g. uploads/traffic.mp4): "
    ).strip().strip('"')

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    # 2. Initialize the three services
    tracker = VehicleTracker()
    counter = VehicleCounter()

    speed_limit_kmh = 50.0
    pixels_per_meter = 10.0  # Demo calibration only

    speed_estimator = SpeedEstimator(
        pixels_per_meter=pixels_per_meter,
        speed_limit_kmh=speed_limit_kmh,
        history_size=5,
    )

    # Use video timestamps for speed calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_number = 0
    window_name = "Vehicle Tracker + Counter + Speed Estimator"

    # OpenCV uses BGR colors
    class_colors = {
        "car": (60, 220, 60),
        "motorcycle": (255, 180, 0),
        "bus": (0, 165, 255),
        "truck": (220, 80, 220),
    }

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 650)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            height, width = frame.shape[:2]

            # 3. Track vehicles and obtain persistent IDs
            tracked_vehicles = tracker.track(frame)

            # 4. Estimate speed using video time
            video_timestamp = frame_number / fps

            vehicles = speed_estimator.estimate(
                tracked_vehicles,
                timestamp=video_timestamp,
            )

            # 5. Count unique tracked vehicles
            count_data = counter.update(vehicles)
            total_counted = count_data["total_vehicles"]
            class_counts = count_data["class_counts"]

            # 6. Draw vehicle boxes and clear labels
            for vehicle in vehicles:
                bbox = vehicle.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = map(int, bbox)

                # Keep coordinates inside the frame
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = str(
                    vehicle.get("class_name", "vehicle")
                )
                track_id = vehicle.get("track_id")
                speed_kmh = vehicle.get("speed_kmh")
                speeding = bool(vehicle.get("speeding", False))

                # Red boundary means estimated speeding
                color = (
                    (0, 0, 255)
                    if speeding
                    else class_colors.get(
                        class_name, (255, 255, 255)
                    )
                )

                if speed_kmh is None:
                    speed_text = "SPEED: CALCULATING"
                else:
                    speed_text = f"SPEED: {speed_kmh:.1f} km/h"

                if track_id is not None:
                    label = (
                        f"{class_name.upper()} #{track_id} "
                        f"| {speed_text}"
                    )
                else:
                    label = (
                        f"{class_name.upper()} | {speed_text}"
                    )

                if speeding:
                    label += " | SPEEDING!"

                # Thick bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3,
                    cv2.LINE_AA,
                )

                # Larger, bold font
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.65
                thickness = 2
                padding_x = 9
                padding_y = 7

                (text_width, text_height), baseline = (
                    cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                )

                label_width = (
                    text_width + padding_x * 2 + 5
                )
                label_height = (
                    text_height + baseline + padding_y * 2
                )

                # Keep the label inside the image
                label_x = max(
                    0,
                    min(x1, max(0, width - label_width)),
                )

                # Prefer above the vehicle
                if y1 >= label_height + 6:
                    label_top = y1 - label_height - 5
                else:
                    # If there is no room above, place inside
                    # the upper part of the vehicle box
                    label_top = min(
                        y1 + 3,
                        max(0, height - label_height - 1),
                    )

                label_bottom = min(
                    height - 1,
                    label_top + label_height,
                )
                label_right = min(
                    width - 1,
                    label_x + label_width,
                )

                # Solid black background for high contrast
                cv2.rectangle(
                    frame,
                    (label_x, label_top),
                    (label_right, label_bottom),
                    (0, 0, 0),
                    -1,
                )

                # Colored stripe matching the vehicle box
                cv2.rectangle(
                    frame,
                    (label_x, label_top),
                    (
                        min(label_x + 5, label_right),
                        label_bottom,
                    ),
                    color,
                    -1,
                )

                text_x = label_x + padding_x + 4
                text_y = min(
                    height - baseline - 1,
                    label_top + padding_y + text_height,
                )

                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            # 7. Display live vehicle counter
            panel_x, panel_y = 12, 12
            panel_width, panel_height = 260, 125

            panel_right = min(
                width - 1, panel_x + panel_width
            )
            panel_bottom = min(
                height - 1, panel_y + panel_height
            )

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_right, panel_bottom),
                (18, 18, 18),
                -1,
            )

            cv2.addWeighted(
                overlay, 0.82, frame, 0.18, 0, frame
            )

            cv2.putText(
                frame,
                f"UNIQUE VEHICLES: {total_counted}",
                (panel_x + 10, panel_y + 23),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            count_lines = [
                f"Cars: {class_counts.get('car', 0)}",
                f"Motorcycles: {class_counts.get('motorcycle', 0)}",
                f"Buses: {class_counts.get('bus', 0)}",
                f"Trucks: {class_counts.get('truck', 0)}",
            ]

            for index, line in enumerate(count_lines):
                cv2.putText(
                    frame,
                    line,
                    (panel_x + 10, panel_y + 46 + index * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )

            # 8. Display demo speed limit
            any_speeding = any(
                vehicle.get("speeding", False)
                for vehicle in vehicles
            )

            cv2.putText(
                frame,
                f"Demo speed limit: {speed_limit_kmh:.0f} km/h",
                (12, max(20, height - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255)
                if any_speeding
                else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Resize only for display
            scale = min(
                1100 / width,
                650 / height,
                1.0,
            )

            display_frame = cv2.resize(
                frame,
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                ),
            )

            cv2.imshow(window_name, display_frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 9. Print final statistics
    final_counts = counter.update([])

    print("\n========== FINAL VEHICLE STATISTICS ==========")
    print(
        f"Total unique vehicles counted: "
        f"{final_counts['total_vehicles']}"
    )

    for vehicle_type, count in final_counts["class_counts"].items():
        print(f"{vehicle_type.capitalize()}: {count}")

    print("===============================================")
    print(f"Demo speed limit: {speed_limit_kmh:.1f} km/h")
    print(
        f"Demo calibration: {pixels_per_meter:.1f} pixels per meter"
    )
    print(
        "WARNING: Speeds are approximate until the camera "
        "and road are calibrated."
    )


if __name__ == "__main__":
    main()
