import traceback
import numpy as np
import time
import os
import threading
import cv2
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient
from constants import SCREENSHOTS_DIR
import scrcpy
from utils import recognize_text, timing_decorator
from cards import CARDS_DATA, ID_TO_CARD, SPELL_CARDS
from logger import Logger

# Load environment variables from .env file
load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10
NUM_CARDS = 4

class ClashRoyaleEnv:
    def __init__(self):
        self.logger = Logger(name="ClashRoyaleEnv")
        self.actions = Actions()
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        # Possible state features
        # - Elixir
        # - Troop presence
        # - Deck hand
        # - Tower states (6 towers)
        self.state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES) + NUM_CARDS + 6

        self.grid_width = 18
        self.grid_height = 28

        self.screenshot_path = os.path.join(SCREENSHOTS_DIR, "current.png")
        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None

        # Tower states tracking (HP normalized 0-1, where 1 = full HP, 0 = destroyed)
        self.tower_states = {
            'enemy_left': 1.0,
            'enemy_king': 1.0, 
            'enemy_right': 1.0,
            'ally_left': 1.0,
            'ally_king': 1.0,
            'ally_right': 1.0
        }
        self.prev_tower_states = self.tower_states.copy()
        
        # Tower HP lookup tables based on levels
        # TODO: Implement it as constants
        self.PRINCESS_TOWER_HP = {
            1: 1400, 2: 1512, 3: 1624, 4: 1750, 5: 1890,
            6: 2030, 7: 2184, 8: 2352, 9: 2534, 10: 2786,
            11: 3052, 12: 3346, 13: 3668, 14: 4032, 15: 4424
        }
        
        self.KING_TOWER_HP = {
            1: 2400, 2: 2568, 3: 2736, 4: 2904, 5: 3096,
            6: 3312, 7: 3528, 8: 3768, 9: 4008, 10: 4392,
            11: 4824, 12: 5304, 13: 5832, 14: 6408, 15: 7032
        }
        
        # Dynamic tower levels and max HP (will be determined from actual HP readings)
        self.tower_levels = {
            'enemy': None,  # Will be detected
            'ally': None    # Will be detected
        }
        
        # Dynamic max HP values (will be set based on detected levels)
        self.tower_max_hp = {
            'enemy_left': None,
            'enemy_king': None,
            'enemy_right': None,
            'ally_left': None,
            'ally_king': None,
            'ally_right': None
        }

        # Tower OCR failure tracking
        self.tower_ocr_failures = {
            'enemy_left': 0,
            'enemy_king': 0, 
            'enemy_right': 0,
            'ally_left': 0,
            'ally_king': 0,
            'ally_right': 0
        }
        
        self.match_over_detected = False

    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
        
        return InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=api_key
        )

    def reset(self):
        self.logger.info("Resetting environment and starting new match")
        self.actions._wait_until_thread_starts()
        self.actions.click_battle_start()
        time.sleep(4)
        self.logger.info("Battle started")
        
        self.game_over_flag = None
        self.match_over_detected = False
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir = None

        self._init_tower_state()
        
        return self._get_state()


    def _init_tower_state(self): 
        # Reset tower states to full HP, 0-1 range
        self.tower_states = {
            'enemy_left': 1.0,
            'enemy_king': 1.0, 
            'enemy_right': 1.0,
            'ally_left': 1.0,
            'ally_king': 1.0,
            'ally_right': 1.0
        }
        
        self.prev_tower_states = self.tower_states.copy()
        
        self.tower_levels = {
            'enemy': None,
            'ally': None
        }

        # Tower max HP values (will be set based on detected levels)
        self.tower_max_hp = {
            'enemy_left': None,
            'enemy_king': None,
            'enemy_right': None,
            'ally_left': None,
            'ally_king': None,
            'ally_right': None
        }
        
        # Reset OCR failure counters
        self.tower_ocr_failures = {
            'enemy_left': 0,
            'enemy_king': 0, 
            'enemy_right': 0,
            'ally_left': 0,
            'ally_king': 0,
            'ally_right': 0
        }
        
        self.logger.extra_visibility(self.check_towers_state())

        self._detect_player_levels_from_ui()

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    @timing_decorator
    def step(self, action_index):
        # If match over, only allow no-op action (last action in list)
        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # No-op action

        # _get_state() should not be called extensively, it requires a lot of processing on the emulator side, which leads to latencies.
        next_state = self._get_state()
        
        if self.game_over_flag:
            done = True
            reward = self._compute_reward(next_state)
            result = self.game_over_flag
            if result == "victory":
                reward += 100
                self.logger.success("Victory detected - ending episode")
            elif result == "defeat":
                reward -= 100
                self.logger.warning("Defeat detected - ending episode")
            return next_state, reward, done

        self.current_cards = self.detect_cards_in_hand()
        self.logger.info(f"Current cards in hand: {self.current_cards}")

        # If all cards are "Unknown", click at center and return no-op
        if all(card == "Unknown" for card in self.current_cards):
            self.logger.warning("All cards are Unknown, clicking at center and skipping move.")
            self.actions._click(640, 400)  # Click at center of screen in device coordinates
            # Return current state, zero reward, not done
            next_state = self._get_state()
            return next_state, 0, False

        action = self.available_actions[action_index]
        self.logger.info(f"Action selected: {action}")

        spell_penalty = 0

        if action["type"] == "PLAY":
            card_index = action["card"]
            x_frac = action["x"]
            y_frac = action["y"]
            if card_index < len(self.current_cards):
                card_name = self.current_cards[card_index]
                self.logger.info(f"Attempting to play {card_name}")
                x = int(x_frac * self.actions.WIDTH)
                y = int(y_frac * self.actions.HEIGHT)
                self.actions.card_play(x, y, card_index)
                # time.sleep(1)  # You can reduce this if needed

                # --- Spell penalty logic ---
                if card_name.lower() in SPELL_CARDS:
                    self.logger.extra_visibility(f"Played spell card: {card_name}")
                    enemy_positions = []
                    for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
                        ex = next_state[i]
                        ey = next_state[i + 1]
                        if ex != 0.0 or ey != 0.0:
                            ex_px = int(ex * self.actions.WIDTH)
                            ey_px = int(ey * self.actions.HEIGHT)
                            enemy_positions.append((ex_px, ey_px))
                    radius = 100
                    found_enemy = any((abs(ex - x) ** 2 + abs(ey - y) ** 2) ** 0.5 < radius for ex, ey in enemy_positions)
                    self.logger.extra_visibility(f"Spell hit enemy: {found_enemy} at positions {enemy_positions}")
                    if not found_enemy:
                        self.logger.warning(f"Spell {card_name} did not hit any enemy, applying penalty.")
                        spell_penalty = -5  # Penalize for wasting spell
                next_state = self._get_state()  # Update state after playing card
            
        # --- Tower damage reward logic ---
        tower_damage_reward = self._compute_tower_damage_reward()

        done = False
        reward = self._compute_reward(next_state) + spell_penalty + tower_damage_reward
        return next_state, reward, done

    @timing_decorator
    def _get_state(self):
        self.actions.capture_area(self.screenshot_path)
        elixir = self.actions.count_elixir()
        
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION environment variable is not set. Please check your .env file.")
        
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )

        # print("RAW results:", results)

        # Handle new structure: dict with "predictions" key
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "predictions" in first:
                predictions = first["predictions"]

        if not predictions:
            self.logger.warning("WARNING: No predictions found in results")
            return None

        # After getting 'predictions' from results:
        if isinstance(predictions, dict) and "predictions" in predictions:
            predictions = predictions["predictions"]

        # for p in predictions:
        #     self.logger.debug(f"{p['class']} at ({p['x']}, {p['y']}) with confidence {p['confidence']:.2f}")
        # self.logger.debug(f"Detected classes: {[repr(p.get('class', '')) for p in predictions if isinstance(p, dict)]}")

        TOWER_CLASSES = {
            "ally king tower",
            "ally princess tower",
            "enemy king tower",
            "enemy princess tower"
        }

        def normalize_class(cls):
            return cls.strip().lower() if isinstance(cls, str) else ""

        allies = [
            (p["x"], p["y"])
            for p in predictions
            if (
                isinstance(p, dict)
                and normalize_class(p.get("class", "")) not in TOWER_CLASSES
                and normalize_class(p.get("class", "")).startswith("ally")
                and "x" in p and "y" in p
            )
        ]

        enemies = [
            (p["x"], p["y"])
            for p in predictions
            if (
                isinstance(p, dict)
                and normalize_class(p.get("class", "")) not in TOWER_CLASSES
                and normalize_class(p.get("class", "")).startswith("enemy")
                and "x" in p and "y" in p
            )
        ]

        self.logger.debug(f"Allies: {allies}")
        self.logger.debug(f"Enemies: {enemies}")

        # Normalize positions
        def normalize(units):
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x, y in units]

        # Pad or truncate to fixed length
        def pad_units(units, max_units):
            units = normalize(units)
            if len(units) < max_units:
                units += [(0.0, 0.0)] * (max_units - len(units))
            return units[:max_units]

        ally_positions = pad_units(allies, MAX_ALLIES)
        enemy_positions = pad_units(enemies, MAX_ENEMIES)

        # Flatten positions
        ally_flat = [coord for pos in ally_positions for coord in pos]
        enemy_flat = [coord for pos in enemy_positions for coord in pos]

        # Add current cards and their elixir costs to state
        card_info = []
        for card_name in self.current_cards:
            card_data = CARDS_DATA.get(card_name.lower(), CARDS_DATA["unknown"])
            card_id = card_data["id"]            
            card_info.extend([card_id])
        
        # Pad with 0 if less than num_cards
        card_info += [0] * (NUM_CARDS - len(card_info))

        # Update tower states
        self.check_towers_state()
        
        # Add tower states to state vector
        tower_states_flat = [
            self.tower_states['enemy_left'],
            self.tower_states['enemy_king'],
            self.tower_states['enemy_right'],
            self.tower_states['ally_left'],
            self.tower_states['ally_king'],
            self.tower_states['ally_right']
        ]

        state = np.array([elixir / 10.0] + ally_flat + enemy_flat + card_info + tower_states_flat, dtype=np.float32)
        return state

    def _compute_reward(self, state):
        if state is None:
            return 0

        elixir = state[0] * 10

        # Extract enemy positions properly (excluding tower positions)
        enemy_positions = []
        for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
            ex = state[i]
            ey = state[i + 1]
            # Only count non-zero positions as enemy presence
            if ex != 0.0 or ey != 0.0:
                enemy_positions.append((ex, ey))
        
        # Calculate enemy presence based on valid enemies only
        enemy_presence = sum(abs(x) + abs(y) for x, y in enemy_positions) if enemy_positions else 0
        
        # Debug output to help diagnose the issue
        self.logger.extra_visibility(f"Enemy valid positions count: {len(enemy_positions)}")
        
        # Base reward on enemy presence but with a lower penalty factor
        # This reduces the strong incentive to do nothing
        reward = -enemy_presence * 0.5  # Reduced penalty factor
        
        # Add a small penalty for no action to encourage the agent to play cards
        if self.prev_elixir is not None and elixir >= 8:
            # If elixir is building up (8 or more), apply an inaction penalty
            reward -= 2.0  # Small penalty for accumulating too much elixir
            self.logger.extra_visibility(f"Applied inaction penalty for high elixir: {elixir}")

        # Elixir efficiency: reward for spending elixir if it reduces enemy presence
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = max(0, self.prev_elixir - elixir)  # Prevent negative values
            enemy_reduced = max(0, self.prev_enemy_presence - enemy_presence)  # Prevent negative values
            
            # Increase the reward for effective elixir spending
            if elixir_spent > 0:
                # Give some reward just for spending elixir (encouraging action)
                reward += elixir_spent * 1.0
                
                # Additional reward if spending elixir coincided with reducing enemies
                if enemy_reduced > 0:
                    reward += 3.0 * min(elixir_spent, enemy_reduced)  # Increased multiplier

        self.prev_elixir = elixir
        self.prev_enemy_presence = enemy_presence

        self.logger.extra_visibility(f"Elixir: {elixir}, Enemy Presence: {enemy_presence}, Reward: {reward}")
        return reward

    def _compute_tower_damage_reward(self):
        """Compute reward based on tower HP changes"""
        # Skip tower damage rewards if max HP values aren't determined yet
        if any(hp is None for hp in self.tower_max_hp.values()):
            self.logger.extra_visibility("Skipping tower damage rewards - max HP values not determined yet")
            return 0
        
        total_reward = 0
        
        # Enemy tower damage (positive reward)
        enemy_towers = ['enemy_left', 'enemy_king', 'enemy_right']
        for tower in enemy_towers:
            prev_hp = self.prev_tower_states[tower]
            curr_hp = self.tower_states[tower]
            damage_dealt = prev_hp - curr_hp
            
            if damage_dealt > 0:
                reward_multiplier = 30
                tower_reward = damage_dealt * reward_multiplier
                total_reward += tower_reward
                
                self.logger.extra_visibility(f"Damage to {tower}: {damage_dealt:.3f} -> Reward: +{tower_reward:.1f}")
                
                # Destruction bonus
                if curr_hp <= 0.01:  # Essentially destroyed
                    destruction_bonus = 50 if tower == 'enemy_king' else 30
                    total_reward += destruction_bonus
                    self.logger.success(f"Destroyed {tower}! Bonus: +{destruction_bonus}")
        
        # Ally tower damage (negative reward - penalty)
        ally_towers = ['ally_left', 'ally_king', 'ally_right']
        for tower in ally_towers:
            prev_hp = self.prev_tower_states[tower]
            curr_hp = self.tower_states[tower]
            damage_taken = prev_hp - curr_hp
            
            if damage_taken > 0:
                penalty_multiplier = 25
                tower_penalty = damage_taken * penalty_multiplier
                total_reward -= tower_penalty
                
                self.logger.warning(f"Damage to our {tower}: {damage_taken:.3f} -> Penalty: -{tower_penalty:.1f}")
                
                # Destruction penalty
                if curr_hp <= 0.01:  # Essentially destroyed
                    destruction_penalty = 60 if tower == 'ally_king' else 40
                    total_reward -= destruction_penalty
                    self.logger.error(f"Lost {tower}! Penalty: -{destruction_penalty}")
        
        if total_reward != 0:
            self.logger.extra_visibility(f"Total tower damage reward: {total_reward:.1f}")
        
        return total_reward

    @timing_decorator
    def detect_cards_in_hand(self):
        try:
            card_paths = self.actions.capture_individual_cards()
            self.logger.debug("\nTesting individual card predictions:")

            cards = []
            workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            if not workspace_name:
                raise ValueError("WORKSPACE_CARD_DETECTION environment variable is not set. Please check your .env file.")
            
            for card_path in card_paths:
                results = self.card_model.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id="custom-workflow",
                    images={"image": card_path}
                )
                # self.logger.debug(f"Card detection raw results: {results}")  # Debug print

                # Fix: parse nested structure
                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])
                if predictions:
                    card_name = predictions[0]["class"]
                    # self.logger.debug(f"Detected card: {card_name}")
                    cards.append(card_name)
                else:
                    self.logger.warning("No card detected.")
                    cards.append("Unknown")
            return cards
        except Exception as e:
            self.logger.error(f"Error in detect_cards_in_hand: {e}")
            return []

    def get_available_actions(self):
        """Generate all possible actions"""
        actions = []
        # Card playing actions
        for card_index in range(NUM_CARDS):
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    actions.append({"type": "PLAY", "card": card_index, "x": x / (self.grid_width - 1), "y": y / (self.grid_height - 1)})
        
        # No-op action
        actions.append({"type": "NO_OP"})
        return actions

    def get_valid_action_mask(self, state):
        """Generate a mask of valid actions based on the current state."""
        mask = np.zeros(self.action_size, dtype=bool)
        current_elixir = state[0] * 10

        # Get card info from state (now comes before tower states)
        card_start_idx = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES)
        card_info_flat = state[card_start_idx:card_start_idx + NUM_CARDS]
        card_ids = [int(card_info_flat[i]) for i in range(len(card_info_flat))]

        for i, action in enumerate(self.available_actions):
            if action["type"] == "NO_OP":
                mask[i] = True
                continue

            card_index = action["card"]
            if card_index < len(card_ids):
                card_id = card_ids[card_index]
                if card_id != 0:  # Card exists
                    card_data = ID_TO_CARD.get(card_id)
                    if card_data and current_elixir >= card_data["elixir"]:
                        mask[i] = True
        return mask

    def _endgame_watcher(self):
        """Thread that watches for both match over and game end conditions"""
        while not self._endgame_thread_stop.is_set():
            # Check for match over first (during game)
            if not self.match_over_detected and hasattr(self.actions, "detect_match_over"):
                if self.actions.detect_match_over():
                    self.logger.info("Match over detected (matchover.png), forcing no-op until next game.")
                    self.match_over_detected = True
            
            # Check for game end (victory/defeat screen)
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            
            # Sleep a bit to avoid hammering the CPU
            time.sleep(0.5)

    def check_towers_state(self):
        """Check and update tower HP states"""
        screenshot = np.array(self.actions.last_frame)
        print(self.actions.last_frame)
        if screenshot is None  or screenshot.size == 0 or not screenshot.any():
            self.logger.warning("Screenshot is empty or invalid.")
            return

        # Tower locations: (x1, y1, x2, y2)
        tower_locations = {
            'enemy_left': (210, 250, 320, 290),
            'enemy_king': (490, 25, 610, 65),
            'enemy_right': (760, 250, 880, 290),
            'ally_left': (210, 1192, 320, 1227),
            'ally_king': (490, 1445, 610, 1490),
            'ally_right': (760, 1192, 880, 1227)
        }
        
        new_tower_states = {}
        
        for tower_name, location in tower_locations.items():
            x1, y1, x2, y2 = location
            img_cropped = screenshot[y1:y2, x1:x2]
            
            try:
                # Ensure the image has the correct format for OCR
                # Convert to RGB if it's not already
                if len(img_cropped.shape) == 3 and img_cropped.shape[2] == 4:
                    # Convert RGBA to RGB
                    img_cropped = img_cropped[:, :, :3]
                    self.logger.debug(f"Tower {tower_name}: Converted RGBA to RGB")
                elif len(img_cropped.shape) == 2:
                    # Convert grayscale to RGB
                    img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2RGB)
                    self.logger.debug(f"Tower {tower_name}: Converted grayscale to RGB")
                elif len(img_cropped.shape) == 3 and img_cropped.shape[2] != 3:
                    # Skip if unexpected number of channels
                    self.logger.warning(f"Tower {tower_name}: Unexpected image shape {img_cropped.shape}, skipping OCR")
                    raise ValueError(f"Unexpected image shape: {img_cropped.shape}")
                
                # Ensure minimum size for OCR
                if img_cropped.shape[0] < 10 or img_cropped.shape[1] < 10:
                    self.logger.warning(f"Tower {tower_name}: Image too small {img_cropped.shape}, skipping OCR")
                    raise ValueError(f"Image too small: {img_cropped.shape}")
                
                results = recognize_text(image=img_cropped)
                
                if len(results) > 0 and results[0]['rec_text'].isnumeric() and float(results[0]['rec_score']) > 0.7:
                    # Successfully recognized HP - reset failure counter
                    current_hp = int(results[0]['rec_text'])
                    max_hp = self.tower_max_hp[tower_name]
                    print(current_hp, max_hp)
                    normalized_hp = min(1.0, max(0.0, current_hp / max_hp))
                    new_tower_states[tower_name] = normalized_hp
                    self.tower_ocr_failures[tower_name] = 0  # Reset failure counter on success
                    self.logger.debug(f"Tower {tower_name}: {current_hp}/{max_hp} HP (normalized: {normalized_hp:.2f})")
                else:
                    # OCR failed - increment failure counter
                    self.tower_ocr_failures[tower_name] += 1
                    
                    # Check if we should mark tower as destroyed after 3+ failures
                    if self.tower_ocr_failures[tower_name] >= 3:
                        new_tower_states[tower_name] = 0.0  # Mark as destroyed
                        self.logger.warning(f"Tower {tower_name}: {self.tower_ocr_failures[tower_name]} consecutive OCR failures - marking as DESTROYED")
                    else:
                        # Could not recognize HP - keep previous state or handle specially
                        if self.tower_max_hp[tower_name] is None:
                            # No max HP determined yet, assume full HP for now
                            new_tower_states[tower_name] = 1.0
                            self.logger.debug(f"Tower {tower_name}: HP unknown, no max HP set yet (assuming full HP)")
                        elif tower_name.endswith('_king') and self.tower_states[tower_name] == 1.0:
                            # King tower might be hidden when not activated - assume full HP
                            new_tower_states[tower_name] = 1.0
                            self.logger.debug(f"Tower {tower_name}: HP hidden (assuming full HP)")
                        else:
                            # Use previous state if recognition failed
                            new_tower_states[tower_name] = self.tower_states[tower_name]
                            self.logger.warning(f"Tower {tower_name}: Could not recognize HP, keeping previous state (failure #{self.tower_ocr_failures[tower_name]})")
                        
            except Exception as e:
                # More graceful error handling for OCR failures
                self.tower_ocr_failures[tower_name] += 1
                
                # Check if we should mark tower as destroyed after 3+ failures
                if self.tower_ocr_failures[tower_name] >= 3:
                    new_tower_states[tower_name] = 0.0  # Mark as destroyed
                    self.logger.warning(f"Tower {tower_name}: {self.tower_ocr_failures[tower_name]} consecutive OCR failures - marking as DESTROYED")
                else:
                    if "AssertionError" in str(e) or "imgC" in str(e):
                        self.logger.debug(f"Tower {tower_name}: OCR format error (image channels), using fallback (failure #{self.tower_ocr_failures[tower_name]})")
                    else:
                        self.logger.warning(f"Tower {tower_name}: OCR error - {str(e)} (failure #{self.tower_ocr_failures[tower_name]})")
                    
                    # Fallback to previous state or default
                    if self.tower_max_hp[tower_name] is None:
                        new_tower_states[tower_name] = 1.0
                        self.logger.debug(f"Tower {tower_name}: Using default full HP due to OCR failure")
                    else:
                        new_tower_states[tower_name] = self.tower_states[tower_name]
                        self.logger.debug(f"Tower {tower_name}: Keeping previous state due to OCR failure")
                
                traceback.print_exc()
        
        # Update tower states
        self.prev_tower_states = self.tower_states.copy()
        self.tower_states = new_tower_states
        
        return new_tower_states

    def _detect_player_levels_from_ui(self):
        """Detect player and opponent tower levels from game UI at match start"""
        screenshot = np.array(self.actions.last_frame)
        
        # UI regions for level detection (x1, y1, x2, y2)
        enemy_level_region = (520, 50, 560, 90)
        ally_level_region = (520, 1433, 560, 1485)
        
        # Extract and process enemy level
        try:
            ex1, ey1, ex2, ey2 = enemy_level_region
            enemy_level_img = screenshot[ey1:ey2, ex1:ex2]
            
            # Ensure proper format for OCR
            if len(enemy_level_img.shape) == 3 and enemy_level_img.shape[2] == 4:
                enemy_level_img = enemy_level_img[:, :, :3]
            elif len(enemy_level_img.shape) == 2:
                enemy_level_img = cv2.cvtColor(enemy_level_img, cv2.COLOR_GRAY2RGB)
            
            enemy_results = recognize_text(image=enemy_level_img)
            
            if len(enemy_results) > 0 and enemy_results[0]['rec_text'].isnumeric() and float(enemy_results[0]['rec_score']) > 0.7:
                enemy_level = int(enemy_results[0]['rec_text'])
                if 1 <= enemy_level <= 15:  # Valid level range
                    self.tower_levels['enemy'] = enemy_level
                    self.logger.info(f"Detected enemy level: {enemy_level}")
                else:
                    self.logger.warning(f"Invalid enemy level detected: {enemy_level}, using default level 9")
                    self.tower_levels['enemy'] = 9
            else:
                self.logger.warning("Could not detect enemy level from UI, using default level 9")
                self.tower_levels['enemy'] = 9
                
        except Exception as e:
            self.logger.error(f"Error detecting enemy level: {e}")
            self.tower_levels['enemy'] = 9
        
        # Extract and process ally level
        try:
            ax1, ay1, ax2, ay2 = ally_level_region
            ally_level_img = screenshot[ay1:ay2, ax1:ax2]
            
            # Ensure proper format for OCR
            if len(ally_level_img.shape) == 3 and ally_level_img.shape[2] == 4:
                ally_level_img = ally_level_img[:, :, :3]
            elif len(ally_level_img.shape) == 2:
                ally_level_img = cv2.cvtColor(ally_level_img, cv2.COLOR_GRAY2RGB)
            
            ally_results = recognize_text(image=ally_level_img)
            
            if len(ally_results) > 0 and ally_results[0]['rec_text'].isnumeric() and float(ally_results[0]['rec_score']) > 0.7:
                ally_level = int(ally_results[0]['rec_text'])
                if 1 <= ally_level <= 15:  # Valid level range
                    self.tower_levels['ally'] = ally_level
                    self.logger.info(f"Detected ally level: {ally_level}")
                else:
                    self.logger.warning(f"Invalid ally level detected: {ally_level}, using default level 9")
                    self.tower_levels['ally'] = 9
            else:
                self.logger.warning("Could not detect ally level from UI, using default level 9")
                self.tower_levels['ally'] = 9
                
        except Exception as e:
            self.logger.error(f"Error detecting ally level: {e}")
            self.tower_levels['ally'] = 9
        
        # Update all tower max HP values based on detected levels
        if self.tower_levels['enemy'] is not None:
            enemy_level = self.tower_levels['enemy']
            self.tower_max_hp['enemy_left'] = self.PRINCESS_TOWER_HP[enemy_level]
            self.tower_max_hp['enemy_right'] = self.PRINCESS_TOWER_HP[enemy_level]
            self.tower_max_hp['enemy_king'] = self.KING_TOWER_HP[enemy_level]
            
        if self.tower_levels['ally'] is not None:
            ally_level = self.tower_levels['ally']
            self.tower_max_hp['ally_left'] = self.PRINCESS_TOWER_HP[ally_level]
            self.tower_max_hp['ally_right'] = self.PRINCESS_TOWER_HP[ally_level]
            self.tower_max_hp['ally_king'] = self.KING_TOWER_HP[ally_level]
        
        self.logger.info(f"Tower levels set - Enemy: {self.tower_levels['enemy']}, Ally: {self.tower_levels['ally']}")
        self.logger.info(f"Tower max HP values updated: {self.tower_max_hp}")
