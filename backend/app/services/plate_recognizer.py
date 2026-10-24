import cv2
import numpy as np
import easyocr


class PlateRecognizer:
    """Recognize text from a detected vehicle license plate."""

    def __init__(self, gpu=False):
        self.reader = easyocr.Reader(["en"], gpu=gpu)

    def recognize(self, plate_image):
        """
        Recognize a license plate from an image.

        Args:
            plate_image: License plate image as a NumPy array.

        Returns:
            Dictionary containing recognized text and confidence.
        """
        if plate_image is None or not isinstance(plate_image, np.ndarray):
            return {
                "text": "",
                "confidence": 0.0,
                "raw_text": "",
            }

        if plate_image.size == 0:
            return {
                "text": "",
                "confidence": 0.0,
                "raw_text": "",
            }

        try:
            # Convert color image to grayscale.
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image.copy()

            # Enlarge small plate images to improve OCR.
            height, width = gray.shape[:2]

            if width < 300 and width > 0:
                scale = 300 / width
                gray = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            # Improve contrast.
            gray = cv2.equalizeHist(gray)

            results = self.reader.readtext(
                gray,
                detail=1,
                paragraph=False,
            )

            if not results:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "raw_text": "",
                }

            raw_text = "".join(
                result[1] for result in results
            )

            # Keep letters and digits, removing spaces and punctuation.
            cleaned_text = "".join(
                char for char in raw_text.upper()
                if char.isalnum()
            )

            confidence = float(
                np.mean([result[2] for result in results])
            )

            return {
                "text": cleaned_text,
                "confidence": confidence,
                "raw_text": raw_text,
            }

        except Exception as exc:
            print(f"License plate OCR error: {exc}")

            return {
                "text": "",
                "confidence": 0.0,
                "raw_text": "",
            }
