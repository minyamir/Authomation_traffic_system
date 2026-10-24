
import cv2
import time

from app.services.vehicle_tracker import VehicleTracker
from app.services.vehicle_counter import VehicleCounter
from app.services.intersection_analyzer import IntersectionAnalyzer
from app.services.traffic_light_recommender import TrafficLightRecommender



class StableTrafficLightController:
    """Hold each signal phase for its full duration and rotate road priority."""

    def __init__(self) -> None:
        self.current_road = None
        self.phase = "ALL_RED"
        self.phase_started = time.monotonic()
        self.phase_duration = 2
        self.green_seconds = 0
        self.current_demand = "LOW"
        self.last_served_road = None

    @staticmethod
    def _green_duration(demand: str, recommendation: dict) -> int:
        minimum = int(recommendation.get("min_green", 10))
        maximum = int(recommendation.get("max_green", 30))
        if maximum < minimum:
            maximum = minimum

        demand = demand.upper()
        if demand == "HIGH":
            return maximum
        if demand == "MODERATE":
            return (minimum + maximum) // 2
        return minimum

    def _ranked_roads(self, recommendation: dict) -> list[dict]:
        ranked = recommendation.get("ranked_roads", [])
        return [
            item for item in ranked
            if isinstance(item, dict) and item.get("road")
        ]

    def _start_green(self, recommendation: dict, now: float) -> None:
        ranked = self._ranked_roads(recommendation)

        if not ranked:
            self.current_road = recommendation.get("recommended_road", "north")
            self.current_demand = recommendation.get("demand", "LOW").upper()
            self.green_seconds = int(recommendation.get("green_seconds", 10))
        else:
            # Prefer the highest-ranked road that did not just receive green.
            candidates = [
                item for item in ranked
                if item["road"] != self.last_served_road
            ]
            selected = (candidates or ranked)[0]
            self.current_road = selected["road"]
            self.current_demand = str(selected.get("demand", "LOW")).upper()
            self.green_seconds = self._green_duration(
                self.current_demand, recommendation
            )

        self.phase = "GREEN"
        self.phase_duration = max(1, self.green_seconds)
        self.phase_started = now

    def update(self, recommendation: dict) -> dict:
        """Update state once per video frame; never change green from demand alone."""
        now = time.monotonic()

        if self.current_road is None:
            self._start_green(recommendation, now)
        elif now - self.phase_started >= self.phase_duration:
            # Change only when the current phase's timer expires.
            if self.phase == "GREEN":
                self.phase = "YELLOW"
                self.phase_duration = max(
                    1, int(recommendation.get("yellow_seconds", 3))
                )
                self.phase_started = now
            elif self.phase == "YELLOW":
                self.last_served_road = self.current_road
                self.phase = "ALL_RED"
                self.phase_duration = max(
                    1, int(recommendation.get("all_red_seconds", 2))
                )
                self.phase_started = now
            else:
                self._start_green(recommendation, now)

        elapsed = max(0.0, now - self.phase_started)
        remaining = max(0, int(self.phase_duration - elapsed + 0.999))

        ranked = self._ranked_roads(recommendation)
        next_road = None
        if ranked:
            next_candidates = [
                item for item in ranked
                if item["road"] != self.current_road
            ]
            if next_candidates:
                next_road = next_candidates[0]["road"]

        return {
            "road": self.current_road,
            "phase": self.phase,
            "remaining_seconds": remaining,
            "green_seconds": self.green_seconds,
            "demand": self.current_demand,
            "next_road": next_road,
        }


def main():
    video_path = input(
        "Enter the path to your traffic video: "
    ).strip().strip('"')

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    tracker = VehicleTracker()
    counter = VehicleCounter()
    intersection_analyzer = IntersectionAnalyzer()
    traffic_light_recommender = TrafficLightRecommender()
    signal_controller = StableTrafficLightController()

    frame_number = 0
    window_name = "Bahir Dar AI - Intelligent Intersection Analysis"

    TEXT_COLOR = (255, 255, 255)
    PANEL_BG = (20, 20, 20)

    CLASS_COLORS = {
        "car": (0, 220, 0),
        "motorcycle": (255, 180, 0),
        "bus": (0, 165, 255),
        "truck": (255, 0, 255),
    }

    ROAD_COLORS = {
        "north": (255, 200, 0),
        "east": (0, 220, 220),
        "south": (255, 0, 255),
        "west": (0, 165, 255),
    }

    CONDITION_COLORS = {
        "LOW": (0, 220, 0),
        "MODERATE": (0, 165, 255),
        "HIGH": (0, 0, 255),
    }

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 650)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            frame_height, frame_width = frame.shape[:2]

            # 1. Detect and track vehicles.
            vehicles = tracker.track(frame)

            # 2. Count unique vehicles across the video.
            count_data = counter.update(vehicles)

            total_counted = count_data["total_vehicles"]
            class_counts = count_data["class_counts"]

            # 3. Define four initial road regions.
            # These proportions assume a centered, cross-shaped
            # intersection. Adjust them to match your actual video.
            road_regions = {
                "north": (
                    int(frame_width * 0.35),
                    0,
                    int(frame_width * 0.65),
                    int(frame_height * 0.40),
                ),
                "east": (
                    int(frame_width * 0.60),
                    int(frame_height * 0.40),
                    frame_width,
                    int(frame_height * 0.60),
                ),
                "south": (
                    int(frame_width * 0.35),
                    int(frame_height * 0.60),
                    int(frame_width * 0.65),
                    frame_height,
                ),
                "west": (
                    0,
                    int(frame_height * 0.40),
                    int(frame_width * 0.40),
                    int(frame_height * 0.60),
                ),
            }

            # 4. Analyze all four approaches independently.
            intersection_data = intersection_analyzer.analyze(
                vehicles,
                road_regions,
            )

            roads = intersection_data["roads"]
            highest_demand = intersection_data["highest_demand"]
            ranking = intersection_data["ranked_approaches"]

            # 5. Recommend a traffic-light phase from the analyzed demand.
            # This is a recommendation only; it does not control real signals.
            light_recommendation = traffic_light_recommender.recommend(
                intersection_data
            )
            recommended_road = light_recommendation["recommended_road"]
            recommended_demand = light_recommendation["demand"]
            green_seconds = light_recommendation["green_seconds"]
            yellow_seconds = light_recommendation["yellow_seconds"]
            all_red_seconds = light_recommendation["all_red_seconds"]

            # The controller keeps the current phase stable until its timer expires.
            signal_state = signal_controller.update(light_recommendation)
            active_road = signal_state["road"]
            signal_phase = signal_state["phase"]
            remaining_seconds = signal_state["remaining_seconds"]
            next_road = signal_state["next_road"]

            # 6. Draw road regions.
            # Thin outlines show which part of the frame is analyzed.
            for road_name, roi in road_regions.items():
                x1, y1, x2, y2 = roi
                road_color = ROAD_COLORS[road_name]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    road_color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    road_name.upper(),
                    (x1 + 8, min(y1 + 25, frame_height - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    road_color,
                    2,
                    cv2.LINE_AA,
                )

            # 7. Draw vehicle bounding boxes and tracking labels.
            for vehicle in vehicles:
                bbox = vehicle.get("bbox")

                if not bbox or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                track_id = vehicle.get("track_id")
                class_name = vehicle.get("class_name", "vehicle")

                box_color = CLASS_COLORS.get(
                    class_name,
                    (255, 255, 255),
                )

                if track_id is not None:
                    label = f"ID {track_id} | {class_name.upper()}"
                else:
                    label = class_name.upper()

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.55
                thickness = 2
                padding = 5

                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness,
                )

                label_width = text_width + padding * 2
                label_height = text_height + baseline + padding * 2

                label_x = max(
                    0,
                    min(x1, frame_width - label_width),
                )

                if y1 >= label_height:
                    label_y = y1 - label_height
                else:
                    label_y = min(
                        y1 + 2,
                        max(0, frame_height - label_height),
                    )

                label_right = min(
                    frame_width - 1,
                    label_x + label_width,
                )
                label_bottom = min(
                    frame_height - 1,
                    label_y + label_height,
                )

                cv2.rectangle(
                    frame,
                    (label_x, label_y),
                    (label_right, label_bottom),
                    PANEL_BG,
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        label_x + padding,
                        label_y + padding + text_height,
                    ),
                    font,
                    font_scale,
                    TEXT_COLOR,
                    thickness,
                    cv2.LINE_AA,
                )

            # 8. Draw the vehicle-count dashboard.
            panel_x = 10
            panel_y = 10
            panel_width = 350
            header_height = 40
            row_height = 32

            panel_height = header_height + 4 * row_height + 35

            cv2.rectangle(
                frame,
                (panel_x, panel_y),
                (
                    min(frame_width - 1, panel_x + panel_width),
                    min(frame_height - 1, panel_y + panel_height),
                ),
                PANEL_BG,
                -1,
            )

            cv2.putText(
                frame,
                "VEHICLE COUNTER",
                (panel_x + 10, panel_y + 27),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            vehicle_types = [
                ("CAR", "car"),
                ("MOTORCYCLE", "motorcycle"),
                ("BUS", "bus"),
                ("TRUCK", "truck"),
            ]

            for index, (label, class_name) in enumerate(vehicle_types):
                row_y = panel_y + header_height + index * row_height
                color = CLASS_COLORS[class_name]

                cv2.putText(
                    frame,
                    label,
                    (panel_x + 12, row_y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    str(class_counts[class_name]),
                    (panel_x + panel_width - 40, row_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            footer_y = panel_y + header_height + 4 * row_height + 23

            cv2.putText(
                frame,
                f"VISIBLE: {len(vehicles)}",
                (panel_x + 12, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"TOTAL: {total_counted}",
                (panel_x + 180, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 9. Draw intersection demand dashboard.
            # Show each road's count, density, and demand level.
            demand_panel_x = 10
            demand_panel_y = panel_y + panel_height + 15
            demand_panel_width = 420
            demand_header_height = 35
            demand_row_height = 35

            demand_panel_height = (
                demand_header_height
                + len(intersection_data["roads"]) * demand_row_height
                + 45
            )

            # Keep the dashboard within the video frame.
            if demand_panel_y + demand_panel_height > frame_height - 5:
                demand_panel_y = max(
                    5,
                    frame_height - demand_panel_height - 5,
                )

            cv2.rectangle(
                frame,
                (demand_panel_x, demand_panel_y),
                (
                    min(
                        frame_width - 1,
                        demand_panel_x + demand_panel_width,
                    ),
                    min(
                        frame_height - 1,
                        demand_panel_y + demand_panel_height,
                    ),
                ),
                PANEL_BG,
                -1,
            )

            cv2.putText(
                frame,
                "INTERSECTION ROAD DEMAND",
                (demand_panel_x + 10, demand_panel_y + 24),
                cv2.FONT_HERSHEY_DUPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for index, road_name in enumerate(
                ("north", "east", "south", "west")
            ):
                data = roads[road_name]
                row_y = (
                    demand_panel_y
                    + demand_header_height
                    + index * demand_row_height
                )

                demand = data["demand"]
                demand_color = CONDITION_COLORS[demand]

                cv2.putText(
                    frame,
                    road_name.upper(),
                    (demand_panel_x + 10, row_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    ROAD_COLORS[road_name],
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Vehicles: {data['vehicles']}",
                    (demand_panel_x + 105, row_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"{data['density_percent']:.1f}%",
                    (demand_panel_x + 245, row_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    demand,
                    (demand_panel_x + 315, row_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    demand_color,
                    2,
                    cv2.LINE_AA,
                )

            # 10. Highlight the highest-demand approach and draw light advice.
            priority_y = (
                demand_panel_y
                + demand_header_height
                + 4 * demand_row_height
                + 30
            )

            priority_text = (
                f"NEXT PRIORITY: {highest_demand.upper()}"
            )

            cv2.putText(
                frame,
                priority_text,
                (demand_panel_x + 10, priority_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.70,
                ROAD_COLORS[highest_demand],
                2,
                cv2.LINE_AA,
            )

            # Show the ranking from highest to lowest demand.
            ranking_text = " > ".join(
                road.upper() for road in ranking
            )

            ranking_y = priority_y + 25

            if ranking_y < frame_height:
                cv2.putText(
                    frame,
                    f"RANK: {ranking_text}",
                    (demand_panel_x + 10, ranking_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

            # 11. Compact, stable traffic-signal status panel.
            # The signal state changes only when its timer expires.
            light_panel_width = min(300, frame_width - 20)
            light_panel_height = 112
            light_panel_x = max(10, frame_width - light_panel_width - 10)
            light_panel_y = 10

            cv2.rectangle(
                frame,
                (light_panel_x, light_panel_y),
                (
                    min(frame_width - 1, light_panel_x + light_panel_width),
                    min(frame_height - 1, light_panel_y + light_panel_height),
                ),
                PANEL_BG,
                -1,
            )

            phase_colors = {
                "GREEN": (0, 220, 0),
                "YELLOW": (0, 220, 255),
                "ALL_RED": (0, 0, 255),
            }
            active_color = phase_colors.get(signal_phase, TEXT_COLOR)

            cv2.putText(
                frame,
                "TRAFFIC SIGNAL",
                (light_panel_x + 9, light_panel_y + 19),
                cv2.FONT_HERSHEY_DUPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            signal_text = (
                "ALL ROADS: RED"
                if signal_phase == "ALL_RED"
                else f"{active_road.upper()}: {signal_phase}"
            )
            cv2.putText(
                frame,
                signal_text,
                (light_panel_x + 9, light_panel_y + 43),
                cv2.FONT_HERSHEY_DUPLEX,
                0.60,
                active_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"TIME LEFT: {remaining_seconds}s",
                (light_panel_x + 9, light_panel_y + 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            next_text = f"NEXT: {next_road.upper()}" if next_road else "NEXT: REASSESS"
            cv2.putText(
                frame,
                next_text,
                (light_panel_x + 9, light_panel_y + 89),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

            # Draw compact signal indicators: active road gets the current phase;
            # every other road remains RED. ALL_RED keeps every road red.
            signal_dot_x = light_panel_x + light_panel_width - 20
            signal_dot_y = light_panel_y + 34
            lamp_color = (
                (0, 220, 0) if signal_phase == "GREEN"
                else (0, 220, 255) if signal_phase == "YELLOW"
                else (0, 0, 255)
            )
            cv2.circle(
                frame,
                (signal_dot_x, signal_dot_y),
                7,
                lamp_color,
                -1,
                cv2.LINE_AA,
            )

            # 12. Resize without changing the aspect ratio.
            scale = min(
                1100 / frame_width,
                650 / frame_height,
                1.0,
            )

            display_frame = cv2.resize(
                frame,
                (
                    max(1, int(frame_width * scale)),
                    max(1, int(frame_height * scale)),
                ),
            )

            cv2.imshow(window_name, display_frame)

            # Press Q to stop playback.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 11. Print final vehicle statistics.
    final_counts = counter.update([])

    print(f"\nProcessed frames: {frame_number}")
    print("\n========== FINAL VEHICLE STATISTICS ==========")
    print(f"Total unique vehicles counted: {final_counts['total_vehicles']}")

    for vehicle_type, count in final_counts["class_counts"].items():
        print(f"{vehicle_type.capitalize()}: {count}")

    print("===============================================")
    print("Traffic-light phases are simulated; no physical signal is controlled.")


if __name__ == "__main__":
    main()
