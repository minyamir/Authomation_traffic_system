
import cv2

from app.services.vehicle_tracker import VehicleTracker


def main():
    video_path = input(
        "Enter the path to your traffic video: "
    ).strip().strip('"')

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    tracker = VehicleTracker()
    frame_number = 0
    window_name = "Bahir Dar AI - Vehicle Tracking"

    # Clean, consistent colors (OpenCV uses BGR).
    BOX_COLOR = (255, 220, 0)       # Cyan
    LABEL_BG = (25, 25, 25)         # Dark background
    TEXT_COLOR = (255, 255, 255)    # White
    PANEL_BG = (25, 25, 25)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 650)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            vehicles = tracker.track(frame)

            frame_height, frame_width = frame.shape[:2]

            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle["bbox"]
                track_id = vehicle["track_id"]
                class_name = vehicle["class_name"]

                label = (
                    f"ID {track_id} | {class_name.upper()}"
                    if track_id is not None
                    else class_name.upper()
                )

                # Draw a clean, thin vehicle boundary.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    BOX_COLOR,
                    2,
                    cv2.LINE_AA,
                )

                # Large, bold, readable label.
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.65
                thickness = 2
                padding = 7

                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness,
                )

                label_width = text_width + padding * 2
                label_height = text_height + baseline + padding * 2

                # Prefer placing the label above the vehicle.
                label_x = max(
                    0,
                    min(x1, frame_width - label_width),
                )

                if y1 >= label_height + 4:
                    label_y = y1 - label_height
                else:
                    # If there's no room above, place it inside the box.
                    label_y = min(
                        y1 + 2,
                        frame_height - label_height,
                    )

                label_y = max(0, label_y)
                label_right = min(
                    frame_width - 1,
                    label_x + label_width,
                )
                label_bottom = min(
                    frame_height - 1,
                    label_y + label_height,
                )

                # Solid dark background behind the text.
                cv2.rectangle(
                    frame,
                    (label_x, label_y),
                    (label_right, label_bottom),
                    LABEL_BG,
                    -1,
                )

                # Cyan border around the label.
                cv2.rectangle(
                    frame,
                    (label_x, label_y),
                    (label_right, label_bottom),
                    BOX_COLOR,
                    1,
                    cv2.LINE_AA,
                )

                # Draw bold white text.
                text_x = label_x + padding
                text_y = min(
                    frame_height - 1,
                    label_y + padding + text_height,
                )

                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    TEXT_COLOR,
                    thickness,
                    cv2.LINE_AA,
                )

            # Compact information panel.
            panel_text = (
                f"VISIBLE VEHICLES: {len(vehicles)}"
                f"   |   FRAME: {frame_number}"
            )

            panel_font = cv2.FONT_HERSHEY_SIMPLEX
            panel_scale = 0.65
            panel_thickness = 2

            (panel_width, panel_height), _ = cv2.getTextSize(
                panel_text,
                panel_font,
                panel_scale,
                panel_thickness,
            )

            cv2.rectangle(
                frame,
                (10, 10),
                (min(frame_width - 1, panel_width + 30), 50),
                PANEL_BG,
                -1,
            )

            cv2.putText(
                frame,
                panel_text,
                (20, 37),
                panel_font,
                panel_scale,
                TEXT_COLOR,
                panel_thickness,
                cv2.LINE_AA,
            )

            # Preserve the video's original aspect ratio.
            height, width = frame.shape[:2]
            scale = min(1100 / width, 650 / height, 1.0)

            display_frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
            )

            cv2.imshow(window_name, display_frame)

            # Press Q to stop playback.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_number} frames.")


if __name__ == "__main__":
    main()
