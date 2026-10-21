
import cv2

from app.services.vehicle_detector import VehicleDetector


def main():
    video_path = input(
        "Enter the path to your traffic video: "
    ).strip().strip('"')

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open the video.")
        print("Check the video path and try again.")
        return

    detector = VehicleDetector()
    frame_number = 0

    window_name = "Bahir Dar Vehicle Detection"

    # Create a smaller, resizable video window.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            detections = detector.detect(frame)

            print(
                f"Frame {frame_number}: "
                f"{len(detections)} vehicles detected"
            )

            for vehicle in detections:
                x1, y1, x2, y2 = vehicle["bbox"]

                label = (
                    f"{vehicle['class_name']} "
                    f"{vehicle['confidence']:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Resize the display only; detection uses the original frame.
            display_frame = cv2.resize(
                frame,
                (960, 540),
            )

            cv2.imshow(window_name, display_frame)

            # Press Q to stop the video.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_number} frames.")


if __name__ == "__main__":
    main()
