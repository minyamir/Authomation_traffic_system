
import cv2


class PlateDetector:
    """
    Detect possible vehicle license plates using OpenCV.

    This is a basic contour-based detector, not a trained
    license plate detection model.
    """

    def __init__(self, min_width=60, min_height=15):
        self.min_width = min_width
        self.min_height = min_height

    def detect(self, frame):
        """
        Detect possible license plates in a video frame.

        Returns a list of dictionaries containing:
        bbox, crop, and confidence.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        frame_height, frame_width = frame.shape[:2]
        candidates = []

        for contour in sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )[:100]:
            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0:
                continue

            polygon = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )

            x, y, width, height = cv2.boundingRect(polygon)

            if width < self.min_width or height < self.min_height:
                continue

            aspect_ratio = width / height

            if not 2.0 <= aspect_ratio <= 6.5:
                continue

            if width > frame_width * 0.8:
                continue

            if height > frame_height * 0.25:
                continue

            crop = frame[y:y + height, x:x + width].copy()

            if crop.size == 0:
                continue

            candidates.append({
                "bbox": (x, y, width, height),
                "crop": crop,
                "confidence": 0.5,
            })

        return candidates

    def detect_plates(self, frame):
        """Compatibility method used by ViolationService."""
        return self.detect(frame)
