"""
Constants for asset paths and configurations.
"""
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Asset directory paths
MAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "main_images")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_LOGS_DIR = os.path.join(MODEL_DIR, "logs")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")

# Main images (UI elements)
BATTLE_START_BUTTON_IMAGE = os.path.join(MAIN_IMAGES_DIR, "battlestartbutton.png")
WINNER_IMAGE = os.path.join(MAIN_IMAGES_DIR, "Winner.png")
OK_BUTTON_IMAGE = os.path.join(MAIN_IMAGES_DIR, "okbutton.png")
MATCH_OVER_IMAGE = os.path.join(MAIN_IMAGES_DIR, "matchover.png")

# Elixir images
ELIXIR_IMAGES = {
    1: os.path.join(MAIN_IMAGES_DIR, "1elixir.png"),
    2: os.path.join(MAIN_IMAGES_DIR, "2elixir.png"),
    3: os.path.join(MAIN_IMAGES_DIR, "3elixir.png"),
    4: os.path.join(MAIN_IMAGES_DIR, "4elixir.png"),
    5: os.path.join(MAIN_IMAGES_DIR, "5elixir.png"),
    6: os.path.join(MAIN_IMAGES_DIR, "6elixir.png"),
    7: os.path.join(MAIN_IMAGES_DIR, "7elixir.png"),
    8: os.path.join(MAIN_IMAGES_DIR, "8elixir.png"),
    9: os.path.join(MAIN_IMAGES_DIR, "9elixir.png"),
    10: os.path.join(MAIN_IMAGES_DIR, "10elixir.png"),
}

# Screenshot paths
CURRENT_SCREENSHOT = os.path.join(SCREENSHOTS_DIR, "current.png")
TEST_SCREENSHOT = os.path.join(SCREENSHOTS_DIR, "test_screenshot.png")
CARDS_SCREENSHOT = os.path.join(SCREENSHOTS_DIR, "cards.png")

# Dynamic screenshot generators
def get_card_screenshot_path(card_number):
    """Generate path for card screenshot (card_1.png, card_2.png, etc.)"""
    return os.path.join(SCREENSHOTS_DIR, f"card_{card_number}.png")

# Model paths
MODELS_LOG_DIR = os.path.join(MODEL_DIR, "logs")

def get_model_path(filename):
    """Generate path for model files"""
    return os.path.join(MODEL_DIR, filename)

def get_model_meta_path(model_filename):
    """Generate meta path for a model file (model_*.pth -> meta_*.json)"""
    meta_filename = model_filename.replace("model_", "meta_").replace(".pth", ".json")
    return os.path.join(MODEL_DIR, meta_filename)
