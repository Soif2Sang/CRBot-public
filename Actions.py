from itertools import count
from adbutils import adb
import io
from PIL import Image
import cv2
import numpy as np
import os
from datetime import datetime
import time
import platform
import traceback
import scrcpy
from utils import recognize_text, take_screenshot, timing_decorator
from logger import Logger
from constants import (
    MAIN_IMAGES_DIR, OK_BUTTON_IMAGE, SCREENSHOTS_DIR, 
    BATTLE_START_BUTTON_IMAGE, WINNER_IMAGE, OK_BUTTON_IMAGE, MATCH_OVER_IMAGE,
    CURRENT_SCREENSHOT, CLOSE_BUTTON_IMAGE, get_card_screenshot_path
)

from utils import take_screenshot, timing_decorator

class Actions:
    def __init__(self):
        self.logger = Logger(name="Actions")
        
        # BlueStacks default resolution (you may need to adjust based on your setup)
        self.device_width = 1080
        self.device_height = 1920
        
        # Define game area coordinates in device space (not screen space)
        # These need to be adjusted based on your BlueStacks resolution and Clash Royale layout
        self.TOP_LEFT_X = 0
        self.TOP_LEFT_Y = 0
        self.BOTTOM_RIGHT_X = self.device_width
        self.BOTTOM_RIGHT_Y = self.device_height
        self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)
    
        self.WIDTH = self.device_height
        self.HEIGHT = self.device_height
        
        # Card bar coordinates in device space
        self.CARD_BAR_X = 237
        self.CARD_BAR_Y = 1590
        self.CARD_BAR_WIDTH = 820
        self.CARD_BAR_HEIGHT = 250
        
        # Initialize ADB connection
        self.device = None
        self.last_frame = None
        self._connect_device()

        # Setup client, which is used to retrieve the video stream (we do not use screencap because it is too slow (average of 300ms per screenshot))
        self.client = scrcpy.Client(
            device=self.device,
            max_width=self.device_height,
            bitrate=1000000000,
            max_fps=10,
        )
        
        self.client.add_listener(scrcpy.EVENT_INIT, self.on_init)
        self.client.add_listener(scrcpy.EVENT_FRAME, self.on_frame)

        # Start the streaming
        self.client.start(threaded=True, daemon_threaded=True)
        self._wait_until_thread_starts()


    def on_init(self):
        self.logger.success(f"Serial: {self.client.device_name}")

    def on_frame(self, frame):
        """Handle incoming video frames"""
        if frame is None:
            return
            return self.logger.info("No frame received")
        else:
            # Convert last_frame from OpenCV BGR to RGB format
            last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            self.last_frame = Image.fromarray(last_frame_rgb)

    def _wait_until_thread_starts(self):
        """Wait until the streaming thread starts"""
        while not self.client.alive or self.last_frame is None:
            time.sleep(0.1)
        else:
            self.logger.info("Streaming thread started")

    def _connect_device(self):
        """Connect to the BlueStacks ADB device"""
        try:
            devices = adb.device_list()
            if not devices:
                self.logger.error("No ADB devices found. Make sure BlueStacks is running and ADB is enabled.")
                self.logger.error("Run setup_adb.py first to configure ADB connection.")
                return False
            
            self.logger.extra_visibility(devices)
            # Usually BlueStacks appears as the first device, but you might need to select the right one
            self.device = devices[0]
            self.logger.success(f"Successfully connected to ADB device: {self.device.serial}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ADB device: {e}")
            self.logger.error("Run setup_adb.py first to configure ADB connection.")
            return False

    @timing_decorator
    def _take_screenshot(self):
        """Take a screenshot using ADB"""
        if not self.device:
            self.logger.warning("No device connected")
            return None
        try:
            # screenshot_bytes = self.device.screencap()
            # screen = Image.open(io.BytesIO(screenshot_bytes))
            screen = take_screenshot(self.device.serial)
            
            # screen.save(CURRENT_SCREENSHOT)
            
            return screen
        except Exception as e:
            self.logger.error(f"Failed to take screenshot: {e}")
            return None

    def _click(self, x, y):
        """Click at coordinates using ADB"""
        if not self.device:
            self.logger.warning("No device connected")
            return False
        
        try:
            self.device.shell(f"input touchscreen swipe {x} {y} {x} {y} 300")
            return True
        except Exception as e:
            self.logger.error(f"Failed to click at ({x}, {y}): {e}")
            return False

    def _swipe(self, x1, y1, x2, y2, duration=500):
        """Swipe from (x1,y1) to (x2,y2) using ADB"""
        if not self.device:
            self.logger.warning("No device connected")
            return False
        
        try:
            self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to swipe: {e}")
            return False

    def capture_area(self, save_path):
        """Capture screenshot of game area using ADB"""
        if self.last_frame is not None:
            self.last_frame.save(save_path)
        else:
            self.logger.warning("Failed to capture screenshot")

    def capture_card_area(self, save_path):
        """Capture screenshot of card area using ADB"""
        if self.last_frame is not None:
            # Crop to card bar area
            cropped = self.last_frame.crop((
                self.CARD_BAR_X, 
                self.CARD_BAR_Y, 
                self.CARD_BAR_X + self.CARD_BAR_WIDTH, 
                self.CARD_BAR_Y + self.CARD_BAR_HEIGHT
            ))
            cropped.save(save_path)
        else:
            self.logger.warning("Failed to capture card area screenshot")

    @timing_decorator
    def capture_individual_cards(self):
        """Capture and split card bar into individual card images using ADB"""
        if self.last_frame is None:
            self.logger.warning("Failed to capture screenshot for individual cards")
            return []
            
        # Crop to card bar area
        card_bar = self.last_frame.crop((
            self.CARD_BAR_X, 
            self.CARD_BAR_Y, 
            self.CARD_BAR_X + self.CARD_BAR_WIDTH, 
            self.CARD_BAR_Y + self.CARD_BAR_HEIGHT
        ))
        
        # Calculate individual card widths
        card_width = self.CARD_BAR_WIDTH // 4
        cards = []
        
        # Split into 4 individual card images
        for i in range(4):
            left = i * card_width
            card_img = card_bar.crop((left, 0, left + card_width, self.CARD_BAR_HEIGHT))
            save_path = get_card_screenshot_path(i+1)
            card_img.save(save_path)
            cards.append(save_path)
        
        return cards

    @timing_decorator
    def count_elixir(self, retry = 0):
        """Count elixir using ADB screenshot analysis"""
        if self.last_frame is None:
            self.logger.warning("Failed to capture screenshot for elixir counting")
            return 0

        if retry == 3:
            return None
        
        image = np.array(self.last_frame)
        img_cropped = image[1825:1887, 290:355]
        
        results = recognize_text(image=img_cropped)
        
        if len(results) > 0 and results[0]['rec_text'].isnumeric() and float(results[0]['rec_score']) > 0.7:
            return max(0, min(int(results[0]['rec_text']), 10))  # Cap at 10 elixir
        else:
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(SCREENSHOTS_DIR, "elixir_count_error.png"))
            self.logger.warning("Failed to recognize elixir count")
        # 5 As default
        return self.count_elixir(retry + 1)

    def _find_template(self, template_path, confidence=0.8, region=None):
        """Find template image in screenshot using OpenCV template matching"""
        if self.last_frame is None:
            return None
            
        # Convert PIL to OpenCV format
        screenshot_cv = cv2.cvtColor(np.array(self.last_frame), cv2.COLOR_RGB2BGR)
        
        # Crop to region if specified
        if region:
            x, y, w, h = region
            screenshot_cv = screenshot_cv[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            offset_x, offset_y = 0, 0
            
        # Load template
        template = cv2.imread(template_path)
        if template is None:
            self.logger.error(f"Could not load template: {template_path}")
            return None
            
        # Perform template matching
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            # Return center coordinates with offset
            template_h, template_w = template.shape[:2]
            center_x = max_loc[0] + template_w // 2 + offset_x
            center_y = max_loc[1] + template_h // 2 + offset_y
            return (center_x, center_y, max_val)
        
        return None

    def card_play(self, x, y, card_index):
        """Play a card using ADB commands"""
        self.logger.info(f"Playing card {card_index} at position ({x}, {y})")
        if card_index in range(4):  # Valid card indices are 0-3
            # Calculate card position in the deck
            card_width = self.CARD_BAR_WIDTH // 4
            card_center_x = self.CARD_BAR_X + (card_index * card_width) + (card_width // 2)
            card_center_y = self.CARD_BAR_Y + (self.CARD_BAR_HEIGHT // 2)
            
            self.logger.debug(f"Clicking on card {card_index} at deck position ({card_center_x}, {card_center_y})")
            self._click(card_center_x, card_center_y)
            time.sleep(0.2)
            
            if y > 1440:
                y = 1440
            if x > 1070:
                x = 1070
                
            self.logger.debug(f"Placing card at battlefield position ({x}, {y})")
            self._click(x, y)
        else:
            self.logger.warning(f"Invalid card index: {card_index} (must be 0-3)")

    def click_battle_start(self):
        """Find and click the battle start button using ADB and template matching"""
        confidences = [0.8, 0.7, 0.6, 0.5]  # Try multiple confidence levels

        # Define the region for the battle button in device coordinates
        battle_button_region = (300, 1350, 480, 300)  # Adjust based on your device resolution

        while True:
            for confidence in confidences:
                self.logger.debug(f"Looking for battle start button (confidence: {confidence})")
                result = self._find_template(BATTLE_START_BUTTON_IMAGE, confidence, battle_button_region)
                if result:
                    x, y, match_confidence = result
                    self._click(x, y)
                    time.sleep(2)
                    return

            # Click at center of screen in device coordinates, could be a chest animation
            self._click(640, 400)              
            # If button not found, click to clear screens
            self.click_ok_button()  # Click OK button if any popups appear

            self.click_close_button()

    def click_close_button(self):
        """Click the close button to close popups"""
        confidences = [0.8, 0.6, 0.4]

        # Define the region for the close button in device coordinates
        close_button_region = (0, 0, self.WIDTH, self.HEIGHT)
        for confidence in confidences:
            self.logger.debug(f"Looking for close button (confidence: {confidence})")
            result = self._find_template(CLOSE_BUTTON_IMAGE, confidence, close_button_region)
            if result:
                x, y, match_confidence = result
                self.logger.info(f"Found close button at ({x}, {y}) with confidence {match_confidence}")
                self._click(x, y)
                time.sleep(2)

    @timing_decorator
    def detect_game_end(self):
        """Detect game end using ADB and template matching"""
        try:

            confidences = [0.8, 0.7, 0.6]

            # Define winner detection region in device coordinates
            winner_region = (0, 0, self.WIDTH, self.HEIGHT)  # Adjust based on your device resolution

            for confidence in confidences:
                # self.logger.debug(f"\nTrying detection with confidence: {confidence}")
                
                result = self._find_template(WINNER_IMAGE, confidence, winner_region)
                if result:
                    x, y, match_confidence = result
                    self.logger.info(f"Found 'Winner' at ({x}, {y}) with confidence {match_confidence}")
                    
                    # Determine if victory or defeat based on position
                    result_type = "victory" if y > 300 else "defeat"  # Adjust threshold based on device
                    self.logger.info(f"Game result: {result_type}")

                    return result_type
        except Exception as e:
            self.logger.error(f"Error in game end detection: {str(e)}")
        return None

    def click_ok_button(self):
        """Click the OK button to close popups"""
        confidences = [0.8, 0.6, 0.4]

        # Define the region for the OK button in device coordinates
        ok_button_region = (0, 0, self.WIDTH, self.HEIGHT)
        for confidence in confidences:
            self.logger.debug(f"Looking for OK button (confidence: {confidence})")
            result = self._find_template(OK_BUTTON_IMAGE, confidence, ok_button_region)
            if result:
                x, y, match_confidence = result
                self.logger.info(f"Found OK button at ({x}, {y}) with confidence {match_confidence}")
                self._click(x, y)
                time.sleep(2)
    
    @timing_decorator
    def detect_match_over(self):
        """Detect match over using ADB and template matching"""
        confidences = [0.8, 0.6, 0.4]
        
        # Define the region where the matchover image appears in device coordinates
        region = (100, 570, 900, 200)  # Adjust based on your device resolution
        
        for confidence in confidences:
            result = self._find_template(MATCH_OVER_IMAGE, confidence, region)
            if result:
                self.logger.info("Match over detected!")
                return True
                
        return False