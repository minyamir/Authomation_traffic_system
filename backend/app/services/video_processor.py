
import cv2

from app.services.vehicle_detector import VehicleDetector


class VideoProcessor:
    """Read a traffic video and detect vehicles in its frames."""

    def __init__(self) -> None:
        self.vehicle_detector = VehicleDetector()

    def process_video(self, video_path: str) -> None:
        """Display a video with detected vehicles."""

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                detections = self.vehicle_detector.detect(frame)

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

                cv2.imshow("Bahir Dar AI - Vehicle Detection", frame)

                # Press Q to stop the video.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
