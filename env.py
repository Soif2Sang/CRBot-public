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

# Reward System Configuration Constants
SIGNIFICANT_DAMAGE_THRESHOLD = 0.1  # 10% of max HP
SPELL_HIT_RADIUS = 100  # pixels
BASE_TOWER_MULTIPLIER = 100
PUSH_BONUS_MULTIPLIER = 150
ELIXIR_EFFICIENCY_MULTIPLIER = 3.0
ENEMY_PRESENCE_FACTOR = 0.5
HIGH_ELIXIR_THRESHOLD = 8
HIGH_ELIXIR_PENALTY = 2.0
SPELL_MISS_PENALTY = 5
VICTORY_BONUS = 100
DEFEAT_PENALTY = 100
KING_TOWER_DESTRUCTION_BONUS = 50
PRINCESS_TOWER_DESTRUCTION_BONUS = 30
KING_TOWER_DESTRUCTION_PENALTY = 60
PRINCESS_TOWER_DESTRUCTION_PENALTY = 40

KING_TOWER_HP = {
    1: 2400, 2: 2568, 3: 2736, 4: 2904, 5: 3096,
    6: 3312, 7: 3528, 8: 3768, 9: 4008, 10: 4392,
    11: 4824, 12: 5304, 13: 5832, 14: 6408, 15: 7032
}

PRINCESS_TOWER_HP = {
    1: 1400, 2: 1512, 3: 1624, 4: 1750, 5: 1890,
    6: 2030, 7: 2184, 8: 2352, 9: 2534, 10: 2786,
    11: 3052, 12: 3346, 13: 3668, 14: 4032, 15: 4424
}

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
        
        self.match_over_detected = False
        self._init_tower_states()

    def _init_tower_states(self):
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
        
        # Track if king towers are awakened (showing HP instead of level)
        self.king_towers_awakened = {
            'enemy_king': False,
            'ally_king': False
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
        time.sleep(5)
        self.logger.info("Battle started")
        
        self.game_over_flag = None
        self.match_over_detected = False
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir = None

        self._init_tower_states()
        self._detect_player_levels_from_ui()
        self.check_towers_state()
        
        return self._get_state()


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
                reward += VICTORY_BONUS
                self.logger.success("Victory detected - ending episode")
            elif result == "defeat":
                reward -= DEFEAT_PENALTY
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
                    found_enemy = any((abs(ex - x) ** 2 + abs(ey - y) ** 2) ** 0.5 < SPELL_HIT_RADIUS for ex, ey in enemy_positions)
                    self.logger.extra_visibility(f"Spell hit enemy: {found_enemy} at positions {enemy_positions}")
                    if not found_enemy:
                        self.logger.warning(f"Spell {card_name} did not hit any enemy, applying penalty.")
                        spell_penalty = -SPELL_MISS_PENALTY
                next_state = self._get_state()  # Update state after playing card
            
        # --- Tower damage reward logic ---
        tower_damage_reward = self._compute_tower_damage_reward()

        done = False
        reward = self._compute_reward(next_state) + spell_penalty + tower_damage_reward
        return next_state, reward, done

    @timing_decorator
    def _get_state(self):
        self.actions.capture_area(self.screenshot_path)
        counted_elixir = self.actions.count_elixir()
        elixir = self.prev_elixir if not counted_elixir else counted_elixir

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
        
        # Store state for push detection
        self._last_state = state
        
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
        reward = -enemy_presence * ENEMY_PRESENCE_FACTOR

        # Add a small penalty for no action to encourage the agent to play cards
        if self.prev_elixir is not None and elixir >= HIGH_ELIXIR_THRESHOLD:
            # If elixir is building up (8 or more), apply an inaction penalty
            reward -= HIGH_ELIXIR_PENALTY
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
                    reward += ELIXIR_EFFICIENCY_MULTIPLIER * min(elixir_spent, enemy_reduced)

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
        
        # Check if we have ally units in enemy territory (upper half of screen)
        ally_units_in_enemy_territory = self._check_ally_units_in_enemy_territory()
        
        # Check if we have enemy units in ally territory (lower half of screen)
        enemy_units_in_ally_territory = self._check_enemy_units_in_ally_territory()
        
        total_reward = 0
        # Enemy tower damage (positive reward)
        enemy_towers = ['enemy_left', 'enemy_king', 'enemy_right']
        for tower in enemy_towers:
            prev_hp = self.prev_tower_states[tower]
            curr_hp = self.tower_states[tower]
            damage_dealt = prev_hp - curr_hp
            
            if damage_dealt > 0:
                tower_reward = damage_dealt * BASE_TOWER_MULTIPLIER
                
                # Bonus reward for significant damage when ally units are pushing
                if damage_dealt > SIGNIFICANT_DAMAGE_THRESHOLD and ally_units_in_enemy_territory:
                    push_bonus = damage_dealt * PUSH_BONUS_MULTIPLIER
                    tower_reward += push_bonus
                    self.logger.success(f"Push damage bonus for {tower}: {damage_dealt:.3f} damage with allies in enemy territory -> Bonus: +{push_bonus:.1f}")
                
                total_reward += tower_reward
                
                self.logger.extra_visibility(f"Damage to {tower}: {damage_dealt:.3f} -> Reward: +{tower_reward:.1f}")
                
                # Destruction bonus
                if curr_hp <= 0.01:  # Essentially destroyed
                    destruction_bonus = KING_TOWER_DESTRUCTION_BONUS if tower == 'enemy_king' else PRINCESS_TOWER_DESTRUCTION_BONUS
                    total_reward += destruction_bonus
                    self.logger.success(f"Destroyed {tower}! Bonus: +{destruction_bonus}")
        
        # Ally tower damage (negative reward - penalty)
        ally_towers = ['ally_left', 'ally_king', 'ally_right']

        self.logger.debug(f"Current tower states: {self.tower_states}")

        
        for tower in ally_towers:
            prev_hp = self.prev_tower_states[tower]
            curr_hp = self.tower_states[tower]
            damage_taken = prev_hp - curr_hp
            
            if damage_taken > 0:
                tower_penalty = damage_taken * BASE_TOWER_MULTIPLIER
                
                # Additional penalty for significant damage when enemy units are pushing
                if damage_taken > SIGNIFICANT_DAMAGE_THRESHOLD and enemy_units_in_ally_territory:
                    push_penalty = damage_taken * PUSH_BONUS_MULTIPLIER
                    tower_penalty += push_penalty
                    self.logger.error(f"Enemy push damage penalty for {tower}: {damage_taken:.3f} damage with enemies in ally territory -> Extra penalty: -{push_penalty:.1f}")
                
                total_reward -= tower_penalty
                
                self.logger.warning(f"Damage to our {tower}: {damage_taken:.3f} -> Penalty: -{tower_penalty:.1f}")
                
                # Destruction penalty
                if curr_hp <= 0.01:  # Essentially destroyed
                    destruction_penalty = KING_TOWER_DESTRUCTION_PENALTY if tower == 'ally_king' else PRINCESS_TOWER_DESTRUCTION_PENALTY
                    total_reward -= destruction_penalty
                    self.logger.error(f"Lost {tower}! Penalty: -{destruction_penalty}")
        
        if total_reward != 0:
            self.logger.success(f"Total tower damage reward: {total_reward:.1f}")
        
        return total_reward

    def _check_ally_units_in_enemy_territory(self):
        """Check if any ally units are in the upper half of the screen (enemy territory)"""
        try:
            # Get the last captured state to access ally positions
            if not hasattr(self, '_last_state') or self._last_state is None:
                return False
            
            state = self._last_state
            
            # Extract ally positions from state
            ally_start_idx = 1  # After elixir
            ally_end_idx = 1 + 2 * MAX_ALLIES
            
            for i in range(ally_start_idx, ally_end_idx, 2):
                ally_x = state[i]
                ally_y = state[i + 1]
                
                # Check if ally unit exists (non-zero position) and is in upper half (y < 0.5)
                if (ally_x != 0.0 or ally_y != 0.0) and ally_y < 0.5:
                    self.logger.extra_visibility("Ally units detected in enemy territory")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking ally units in enemy territory: {e}")
            return False

    def _check_enemy_units_in_ally_territory(self):
        """Check if any enemy units are in the lower half of the screen (ally territory)"""
        try:
            # Get the last captured state to access enemy positions
            if not hasattr(self, '_last_state') or self._last_state is None:
                return False
            
            state = self._last_state
            
            # Extract enemy positions from state
            enemy_start_idx = 1 + 2 * MAX_ALLIES  # After elixir and ally positions
            enemy_end_idx = 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES
            
            for i in range(enemy_start_idx, enemy_end_idx, 2):
                enemy_x = state[i]
                enemy_y = state[i + 1]
                
                # Check if enemy unit exists (non-zero position) and is in lower half (y > 0.5)
                if (enemy_x != 0.0 or enemy_y != 0.0) and enemy_y > 0.5:
                    self.logger.warning("Enemy units detected in ally territory")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking enemy units in ally territory: {e}")
            return False

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
                
                time.sleep(5)
                    
                self.actions.click_ok_button()
                break
            
            # Sleep a bit to avoid hammering the CPU
            time.sleep(1)

    def check_towers_state(self):
        """Check and update tower HP states"""
        screenshot = np.array(self.actions.last_frame)

        if screenshot is None  or screenshot.size == 0 or not screenshot.any():
            self.logger.warning("Screenshot is empty or invalid.")
            return

        # Tower locations: (x1, y1, x2, y2)
        tower_locations = {
            'enemy_left': (210, 250, 320, 290),
            'enemy_right': (760, 250, 880, 290),
            'ally_left': (210, 1192, 320, 1227),
            'ally_right': (760, 1192, 880, 1227),
            'enemy_king': (490, 25, 610, 65),
            'ally_king': (490, 1445, 610, 1490),
        }
        
        new_tower_states = self.tower_states.copy()
        
        for tower_name, location in tower_locations.items():
            x1, y1, x2, y2 = location
            img_cropped = screenshot[y1:y2, x1:x2]
            
            try:
                results = recognize_text(image=img_cropped)
                
                if len(results) > 0 and results[0]['rec_text'].isnumeric() and float(results[0]['rec_score']) > 0.7:
                    current_hp = int(results[0]['rec_text'])
                    max_hp = self.tower_max_hp[tower_name]
                    
                    # Special handling for king towers awakening
                    if '_king' in tower_name:
                        # Check if this king tower just awakened
                        if not self.king_towers_awakened[tower_name]:
                            # King tower is awakening - check if HP reading is close to max HP
                            if max_hp and abs(current_hp - max_hp) <= 100:  # Within 100 HP of max
                                self.logger.info(f"King tower {tower_name} awakened with HP: {current_hp}")
                                self.king_towers_awakened[tower_name] = True
                                # Initialize with current HP instead of treating as damage
                                normalized_hp = min(1.0, max(0.0, current_hp / max_hp))
                                new_tower_states[tower_name] = normalized_hp
                                # Update previous state to prevent false damage calculation
                                self.prev_tower_states[tower_name] = normalized_hp
                                continue
                    
                    # Normal HP processing for awakened towers or princess towers
                    if max_hp:
                        normalized_hp = min(1.0, max(0.0, current_hp / max_hp))
                        
                        # Validate HP change to prevent unrealistic jumps
                        prev_hp = self.tower_states[tower_name]
                        hp_change = abs(normalized_hp - prev_hp)
                        
                        # If HP change is too large (>50% in one frame), it's likely an OCR error or awakening
                        if hp_change > 0.5 and '_king' in tower_name and not self.king_towers_awakened[tower_name]:
                            self.logger.warning(f"Large HP change detected for {tower_name}: {prev_hp:.2f} -> {normalized_hp:.2f}, treating as awakening")
                            self.king_towers_awakened[tower_name] = True
                            # Update previous state to prevent false damage calculation
                            self.prev_tower_states[tower_name] = normalized_hp
                        
                        new_tower_states[tower_name] = normalized_hp
                        self.logger.debug(f"Tower {tower_name}: {current_hp}/{max_hp} HP (normalized: {normalized_hp:.2f})")
                    
                elif '_king' in tower_name:
                    # Special handling for king towers showing level instead of HP
                    if not self.king_towers_awakened[tower_name]:
                        # King tower still showing level, not awakened
                        if 'enemy' in tower_name and self.tower_states['enemy_left'] == 1.0 and self.tower_states['enemy_right'] == 1.0:
                            new_tower_states[tower_name] = 1.0  # Keep at full HP
                        elif 'ally' in tower_name and self.tower_states['ally_left'] == 1.0 and self.tower_states['ally_right'] == 1.0:
                            new_tower_states[tower_name] = 1.0  # Keep at full HP
                    
            except Exception as e:
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
            self.tower_max_hp['enemy_left'] = PRINCESS_TOWER_HP[enemy_level]
            self.tower_max_hp['enemy_right'] = PRINCESS_TOWER_HP[enemy_level]
            self.tower_max_hp['enemy_king'] = KING_TOWER_HP[enemy_level]
            
        if self.tower_levels['ally'] is not None:
            ally_level = self.tower_levels['ally']
            self.tower_max_hp['ally_left'] = PRINCESS_TOWER_HP[ally_level]
            self.tower_max_hp['ally_right'] = PRINCESS_TOWER_HP[ally_level]
            self.tower_max_hp['ally_king'] = KING_TOWER_HP[ally_level]
        
        self.logger.info(f"Tower levels set - Enemy: {self.tower_levels['enemy']}, Ally: {self.tower_levels['ally']}")
        self.logger.info(f"Tower max HP values updated: {self.tower_max_hp}")
