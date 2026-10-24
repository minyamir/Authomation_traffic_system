
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

from app.services.plate_detector import PlateDetector
from app.services.plate_recognizer import PlateRecognizer


def process_plates(frame, detector, recognizer):
    """Run plate detection and OCR outside the display loop."""
    candidates = detector.detect_plates(frame)

    # Limit OCR workload to the two largest candidates.
    candidates = sorted(
        candidates,
        key=lambda item: item["bbox"][2] * item["bbox"][3],
        reverse=True,
    )[:2]

    results = []

    for candidate in candidates:
        result = recognizer.recognize(candidate["crop"])

        results.append({
            "bbox": candidate["bbox"],
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
        })

    return results


def run_license_plate_test():
    video_path = input(
        "Enter video path (e.g. uploads/License.mp4): "
    ).strip().strip('"')

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0

    frame_delay_ms = max(1, int(1000 / fps))

    detector = PlateDetector()
    recognizer = PlateRecognizer(gpu=False)

    # Only one background task runs at a time.
    executor = ThreadPoolExecutor(max_workers=1)
    pending_task = None

    frame_number = 0
    last_results = []
    readable_plates = set()
    submitted_frames = 0

    # Run detection every 10 frames, if the worker is available.
    detection_interval = 10

    window_name = "Bahir Dar AI - License Plate Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print("\nVideo started. Press Q to quit.")
    print("Plate recognition runs in the background.")

    try:
        while True:
            loop_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1

            # Collect completed background work without blocking.
            if pending_task is not None and pending_task.done():
                try:
                    last_results = pending_task.result()

                    for item in last_results:
                        if item["text"]:
                            readable_plates.add(item["text"])
                            print(
                                f"Frame result: {item['text']} "
                                f"(confidence {item['confidence']:.2f})"
                            )
                except Exception as exc:
                    print(f"Plate processing error: {exc}")

                pending_task = None

            # Submit a frame only if the previous task has finished.
            if (
                frame_number % detection_interval == 0
                and pending_task is None
            ):
                pending_task = executor.submit(
                    process_plates,
                    frame.copy(),
                    detector,
                    recognizer,
                )
                submitted_frames += 1

            # Display the video continuously.
            height, width = frame.shape[:2]
            display_width = min(960, width)
            display_height = max(
                1, int(height * display_width / width)
            )

            display_frame = cv2.resize(
                frame,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )

            # Show the latest OCR result as text.
            # Results may refer to an earlier frame.
            y_text = 30

            for item in last_results:
                label = (
                    f"Plate: {item['text'] or 'Unreadable'} "
                    f"({item['confidence']:.2f})"
                )

                cv2.putText(
                    display_frame,
                    label,
                    (15, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                y_text += 28

            cv2.putText(
                display_frame,
                f"Frame: {frame_number}",
                (15, display_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.imshow(window_name, display_frame)

            # Keep playback moving at approximately the video's FPS.
            elapsed_ms = (time.perf_counter() - loop_start) * 1000
            wait_ms = max(1, int(frame_delay_ms - elapsed_ms))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Do not make the window wait for a slow OCR job.
        executor.shutdown(wait=False, cancel_futures=True)

    print("\n========== TEST RESULTS ==========")
    print(f"Frames displayed: {frame_number}")
    print(f"Frames submitted for recognition: {submitted_frames}")
    print(f"Distinct recognized texts: {len(readable_plates)}")
    print(f"Recognized text: {sorted(readable_plates)}")
    print("==================================")


if __name__ == "__main__":
    run_license_plate_test()
