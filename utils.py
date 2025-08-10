from functools import wraps
import subprocess
import time
from PIL import Image
from setup_adb import find_adb_executable
from paddleocr import TextRecognition, PaddleOCR  
import numpy as np

TEXT_RECOGNITION_MODEL = TextRecognition()
ADB_PATH = find_adb_executable()
DEBUGGING = False

def timing_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        if DEBUGGING:
            # Use class logger if available, otherwise fall back to print
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"[Performance] {func.__name__:<25} took {execution_time:.4f} seconds")
            else:
                print(f"[Performance] {func.__name__:<25} took {execution_time:.4f} seconds")

        return result
    return wrapper

def recognize_text(image):
    """Recognize text in the given image using OCR."""
    # Use the text recognition model to extract text
    results = TEXT_RECOGNITION_MODEL.predict(image)

    # print(results)

    # for result in results:
        # print(f"Detected text: {result['rec_text']} (confidence: {result['rec_score']})")

    return results
