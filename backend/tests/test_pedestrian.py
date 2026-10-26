import cv2

from app.services.pedestrian_detector import PedestrianDetector
from app.services.pedestrian_tracker import PedestrianTracker
from app.services.pedestrian_counter import PedestrianCounter
from app.services.crossing_controller import CrossingController


VIDEO_PATH = "uploads/Mans.mp4"

# Adjust these coordinates to match the pedestrian crossing in your video.
CROSSING_ZONE = (200, 200, 300, 150)

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 650

# OpenCV uses BGR color order.
GREEN = (0, 220, 0)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
BLUE = (255, 100, 0)
ORANGE = (0, 140, 255)
GRAY = (170, 170, 170)


def draw_label(frame, text, x, y, color, scale=0.6, thickness=2):
    """Draw readable text with a solid background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    x = max(0, min(x, frame.shape[1] - text_width - 8))
    y = max(text_height + 8, min(y, frame.shape[0] - baseline - 4))

    cv2.rectangle(
        frame,
        (x, y - text_height - 8),
        (x + text_width + 8, y + baseline - 2),
        BLACK,
        cv2.FILLED,
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 3),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_pedestrian(frame, person):
    """Draw a distinct bounding box and readable ID for each person."""
    x, y, width, height = person["bbox"]
    track_id = person["track_id"]
    confidence = person.get("confidence", 0.0)

    # Rotate through clearly different colors for different tracking IDs.
    palette = [
        (0, 255, 0),       # Green
        (255, 180, 0),     # Blue/orange
        (0, 200, 255),     # Yellow-orange
        (255, 0, 255),     # Magenta
        (255, 255, 0),     # Cyan
        (0, 100, 255),     # Orange-red
        (180, 255, 0),     # Lime
        (255, 100, 180),   # Pink
    ]
    color = palette[(track_id - 1) % len(palette)]

    x2 = x + width
    y2 = y + height

    # Thick outer white border, then a colored inner border.
    cv2.rectangle(frame, (x, y), (x2, y2), WHITE, 4)
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    draw_label(
        frame,
        f"PERSON ID {track_id} | {confidence:.0%}",
        x,
        max(30, y - 5),
        color,
        scale=0.58,
        thickness=2,
    )


def draw_crossing_zone(frame, zone):
    """Highlight the crossing without covering the people."""
    x, y, width, height = zone
    x2 = x + width
    y2 = y + height

    # Bright, thick border with a contrasting outer edge.
    cv2.rectangle(frame, (x, y), (x2, y2), BLACK, 6)
    cv2.rectangle(frame, (x, y), (x2, y2), CYAN, 3)

    # Corner markers make the zone easier to identify.
    corner = 24
    for cx, cy, dx, dy in [
        (x, y, 1, 1),
        (x2, y, -1, 1),
        (x, y2, 1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(
            frame,
            (cx, cy),
            (cx + dx * corner, cy),
            WHITE,
            4,
        )
        cv2.line(
            frame,
            (cx, cy),
            (cx, cy + dy * corner),
            WHITE,
            4,
        )

    draw_label(
        frame,
        "PEDESTRIAN CROSSING ZONE",
        x,
        max(35, y - 8),
        CYAN,
        scale=0.65,
        thickness=2,
    )


def draw_information_panel(frame, pedestrians, count_result, signal_status, frame_number):
    """Draw a high-contrast status panel over the video."""
    panel_width = min(420, frame.shape[1] - 20)
    panel_height = 235

    # Slightly transparent black panel.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        BLACK,
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    x = 25
    y = 42

    cv2.putText(
        frame,
        "BAHIR DAR AI TRAFFIC",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        CYAN,
        2,
        cv2.LINE_AA,
    )

    y += 34
    cv2.putText(
        frame,
        "PEDESTRIAN MONITOR",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    y += 32
    cv2.putText(
        frame,
        f"People detected: {len(pedestrians)}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    y += 29
    cv2.putText(
        frame,
        f"People in zone: {count_result['people_inside']}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        CYAN,
        2,
        cv2.LINE_AA,
    )

    y += 29
    cv2.putText(
        frame,
        f"Unique entries: {count_result['total_crossings']}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    y += 31
    vehicle_signal = signal_status["vehicle_signal"]
    pedestrian_signal = signal_status["pedestrian_signal"]

    vehicle_color = {
        "RED": RED,
        "YELLOW": YELLOW,
        "GREEN": GREEN,
    }.get(vehicle_signal, WHITE)

    pedestrian_color = GREEN if pedestrian_signal == "WALK" else RED

    cv2.putText(
        frame,
        f"VEHICLE: {vehicle_signal}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        vehicle_color,
        2,
        cv2.LINE_AA,
    )

    y += 30
    cv2.putText(
        frame,
        f"PEDESTRIAN: {pedestrian_signal}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        pedestrian_color,
        2,
        cv2.LINE_AA,
    )

    # Frame number in the top-right corner.
    draw_label(
        frame,
        f"FRAME {frame_number}",
        frame.shape[1] - 160,
        35,
        WHITE,
        scale=0.6,
    )


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    detector = PedestrianDetector()
    tracker = PedestrianTracker(detector=detector)

    counter = PedestrianCounter(
        crossing_zone=CROSSING_ZONE
    )
    controller = CrossingController()

    window_name = "Bahir Dar AI - Clear Pedestrian Monitoring"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
    )

    frame_number = 0

    print("Starting pedestrian detection and tracking.")
    print("Press Q to quit the video window.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            # 1. Detect and track pedestrians.
            pedestrians = tracker.update(frame)

            # 2. Count unique tracked people entering the zone.
            count_result = counter.update(pedestrians)

            # 3. Update the simulated crossing signals.
            signal_status = controller.update(
                pedestrians_waiting=count_result["people_inside"]
            )

            # 4. Draw the crossing zone first.
            annotated = frame.copy()
            draw_crossing_zone(annotated, CROSSING_ZONE)

            # 5. Draw pedestrian boxes and labels on top of the zone.
            for person in pedestrians:
                draw_pedestrian(annotated, person)

            # 6. Draw the high-contrast information panel last.
            draw_information_panel(
                annotated,
                pedestrians,
                count_result,
                signal_status,
                frame_number,
            )

            # Resize while preserving the video's original aspect ratio.
            height, width = annotated.shape[:2]
            scale = min(
                DISPLAY_WIDTH / width,
                DISPLAY_HEIGHT / height,
            )

            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            display = cv2.resize(
                annotated,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(window_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nPedestrian test completed.")
    print(f"Frames processed: {frame_number}")
    print(f"Unique track IDs entering the zone: {counter.total_crossings}")


if __name__ == "__main__":
    main()
    