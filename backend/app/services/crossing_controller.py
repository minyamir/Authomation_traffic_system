import time
from typing import Any


class CrossingController:
    """Simulate coordinated pedestrian and vehicle signal states."""

    def __init__(
        self,
        yellow_seconds: float = 3.0,
        walk_seconds: float = 8.0,
        clearance_seconds: float = 3.0,
        clock: Any = time.monotonic,
    ) -> None:
        if min(
            yellow_seconds,
            walk_seconds,
            clearance_seconds,
        ) < 0:
            raise ValueError("Signal timings cannot be negative")

        if walk_seconds == 0:
            raise ValueError("walk_seconds must be greater than zero")

        self.yellow_seconds = yellow_seconds
        self.walk_seconds = walk_seconds
        self.clearance_seconds = clearance_seconds
        self.clock = clock

        self.vehicle_signal = "GREEN"
        self.pedestrian_signal = "DONT_WALK"

        self.state = "VEHICLE_GREEN"
        self.state_started_at = self.clock()
        self.pedestrian_request = False

    def request_crossing(self) -> dict[str, Any]:
        """Register a pedestrian request for the next safe crossing."""
        self.pedestrian_request = True
        self._advance()
        return self.get_status()

    def _set_state(self, state: str) -> None:
        self.state = state
        self.state_started_at = self.clock()

        if state == "VEHICLE_GREEN":
            self.vehicle_signal = "GREEN"
            self.pedestrian_signal = "DONT_WALK"

        elif state == "VEHICLE_YELLOW":
            self.vehicle_signal = "YELLOW"
            self.pedestrian_signal = "DONT_WALK"

        elif state == "PEDESTRIAN_WALK":
            self.vehicle_signal = "RED"
            self.pedestrian_signal = "WALK"

        elif state == "PEDESTRIAN_CLEARANCE":
            self.vehicle_signal = "RED"
            self.pedestrian_signal = "DONT_WALK"

    def _advance(self) -> None:
        """Advance the state machine when its timer expires."""
        now = self.clock()
        elapsed = now - self.state_started_at

        if (
            self.state == "VEHICLE_GREEN"
            and self.pedestrian_request
        ):
            self._set_state("VEHICLE_YELLOW")

        elif (
            self.state == "VEHICLE_YELLOW"
            and elapsed >= self.yellow_seconds
        ):
            self._set_state("PEDESTRIAN_WALK")

        elif (
            self.state == "PEDESTRIAN_WALK"
            and elapsed >= self.walk_seconds
        ):
            self._set_state("PEDESTRIAN_CLEARANCE")

        elif (
            self.state == "PEDESTRIAN_CLEARANCE"
            and elapsed >= self.clearance_seconds
        ):
            self.pedestrian_request = False
            self._set_state("VEHICLE_GREEN")

    def update(
        self,
        pedestrians_waiting: int = 0,
    ) -> dict[str, Any]:
        """Update the simulation using the current pedestrian count."""
        if pedestrians_waiting < 0:
            raise ValueError(
                "pedestrians_waiting cannot be negative"
            )

        if pedestrians_waiting > 0:
            self.pedestrian_request = True

        self._advance()
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        """Return the current signal state and elapsed time."""
        elapsed = max(0.0, self.clock() - self.state_started_at)

        return {
            "state": self.state,
            "vehicle_signal": self.vehicle_signal,
            "pedestrian_signal": self.pedestrian_signal,
            "pedestrian_request_pending": self.pedestrian_request,
            "state_elapsed_seconds": round(elapsed, 2),
        }