import time
import random
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

# Assuming Control class is defined elsewhere and has press, release, press_for methods
from roblox.control import Control 
import json
import os

class NavigationSystem:
    """Manages all character movement, including target approach, anti-stuck, and search patterns."""

    def __init__(self, config: Dict[str, Any], control: Control):
        """
        Initializes the NavigationSystem.

        Args:
            config: The main configuration dictionary for the bot.
            control: An instance of the Control class for sending keyboard/mouse inputs.
        """
        self.config: Dict[str, Any] = config
        self.control: Control = control
        
        # Movement State
        self.smoothed_cx: Optional[float] = None
        self.smoothed_cy: Optional[float] = None
        self.previous_distance: float = float('inf')
        self.stuck_frames: int = 0
        self.navigation_phase: str = 'APPROACHING' # Start by approaching
        self.aim_completed_frames: int = 0
        
        # Anti-Stuck and Escape State
        self.last_escape_time: float = 0.0
        self.escape_stage: int = 0
        
        # Confirmation State
        self.confirmation_start_time: float = 0.0
        self.confirmation_failures: int = 0

        # Search State
        self.search_phase: str = 'SCANNING'
        self.search_scan_start_time: float = 0.0
        self.search_move_start_time: float = 0.0
        self.search_turn_direction: str = 'turn_right'
        self.current_search_pattern: int = 0

        # Short-term memory
        self.short_term_memory: List[Dict[str, Any]] = []
        
        # Long-term memory (Map Cache)
        self.map_cache_path = self.config['memory'].get('map_cache_path', 'map_cache.json')
        self.long_term_memory = self._load_map_cache()

    def _load_map_cache(self) -> Dict[str, Any]:
        """Loads the map cache from a file."""
        if os.path.exists(self.map_cache_path):
            try:
                with open(self.map_cache_path, 'r') as f:
                    print(f"[MEMORY] Загрузка кэша карты из {self.map_cache_path}")
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[ERROR] Не удалось загрузить кэш карты: {e}")
        return {"bad_spots": []}

    def _save_map_cache(self):
        """Saves the current map cache to a file."""
        try:
            with open(self.map_cache_path, 'w') as f:
                json.dump(self.long_term_memory, f, indent=4)
        except IOError as e:
            print(f"[ERROR] Не удалось сохранить кэш карты: {e}")

    def reset_movement_state(self):
        """Resets all movement-related state variables to their defaults."""
        self.smoothed_cx = None
        self.smoothed_cy = None
        self.previous_distance = float('inf')
        self.stuck_frames = 0
        self.navigation_phase = 'APPROACHING'
        self.aim_completed_frames = 0
        self.escape_stage = 0

    def move_to_target(self, target: Dict[str, Any], frame_shape: Tuple[int, int, int], fps: float) -> Tuple[str, Optional[float]]:
        """
        Calculates and executes movement towards a target for a single frame.

        Args:
            target: The dictionary representing the current target.
            frame_shape: The shape of the video frame.
            fps: The current frames per second of the bot.

        Returns:
            A tuple containing the navigation status ('at_target', 'stuck', 'navigating')
            and the horizontal error if stuck.
        """
        frame_h, frame_w, _ = frame_shape
        center_x, center_y = frame_w / 2, frame_h / 2

        cx, cy = target['cx'], target['cy']
        
        # Apply EMA smoothing for more stable movement
        if self.smoothed_cx is None:
            self.smoothed_cx, self.smoothed_cy = cx, cy
        else:
            dynamic_alpha = self.config['movement']['aim_alpha'] * (1.0 + 0.3 * (1.0 - min(1.0, target['distance'] / 300.0)))
            self.smoothed_cx = dynamic_alpha * cx + (1 - dynamic_alpha) * self.smoothed_cx
            self.smoothed_cy = dynamic_alpha * cy + (1 - dynamic_alpha) * self.smoothed_cy
        
        error_x = self.smoothed_cx - center_x
        error_y = self.smoothed_cy - center_y
        distance = target['distance']

        # Check if target is reached
        if distance < self.config['movement']['min_distance']:
            return 'at_target', None

        # --- Anti-stuck Logic ---
        progress_made = self.previous_distance is None or (self.previous_distance - distance) > 3.0
        if not progress_made:
            self.stuck_frames += 1
        else:
            self.stuck_frames = max(0, self.stuck_frames - 1)

        # Check for severe stuck situation
        if self.stuck_frames >= self.config['antistuck']['stuck_detection_threshold']:
            self.escape_stage = 0
            return 'stuck', error_x
        
        # Check for minor obstacle and attempt to avoid
        if self.config['antistuck']['stuck_for_obstacle_avoidance_threshold'] < self.stuck_frames:
            self._perform_obstacle_avoidance(error_x)
            return 'navigating', None

        # --- Navigation Logic ---
        self._approach_target(error_x, error_y, distance, fps)
        
        self.previous_distance = distance
        return 'navigating', None

    def _aim_at_target(self, error_x: float, frame_shape: Tuple[int, int, int]):
        """Handles the aiming phase of navigation."""
        self.control.release_all_keys()
        
        if abs(error_x) > self.config['navigation']['aim_complete_threshold_px']:
            self.aim_completed_frames = 0
            turn_key = 'turn_left' if error_x < 0 else 'turn_right'
            turn_duration = self._calculate_turn_duration(error_x, frame_shape)
            self.control.press_for(turn_key, turn_duration)
        else:
            self.aim_completed_frames += 1

        if self.aim_completed_frames >= self.config['dynamic_turn']['aim_confirm_threshold']:
            self.navigation_phase = 'APPROACHING'
            self.aim_completed_frames = 0

    def _approach_target(self, error_x: float, error_y: float, distance: float, fps: float):
        """Handles the approaching phase of navigation."""
        self.control.release_all_keys()
        self.control.press('up')

        # Proportional turning control while moving forward
        if abs(error_x) > self.config['navigation']['aim_complete_threshold_px']:
            turn_key = 'turn_left' if error_x < 0 else 'turn_right'
            self.control.press(turn_key)
        
        if error_y < -50 and distance < 200 and random.random() < 0.3:
            self.control.press('jump')

        # Dynamically adjust sleep time based on FPS and distance for smoother movement
        fps_factor = max(1.0, fps / 30.0)
        distance_factor = min(1.2, 1.0 + (distance / 500.0) * 0.2)
        sleep_duration = (self.config['movement']['forward_max_step'] / fps_factor) * distance_factor
        time.sleep(float(sleep_duration))

    def _perform_obstacle_avoidance(self, error_x: float):
        """Performs a small maneuver to get around a minor obstacle."""
        self.control.release_all_keys()
        strafe_key = 'left' if error_x > 0 else 'right'
        turn_key = 'turn_right' if error_x > 0 else 'turn_left'
        self.control.press('up')
        self.control.press(strafe_key)
        self.control.press(turn_key)
        time.sleep(0.35)
        self.control.release_all_keys() # Release all to be safe
        self.stuck_frames += 1

    def perform_escape_maneuver(self, error: float):
        """Executes a multi-stage maneuver to escape a stuck position."""
        now = time.time()
        if now - self.last_escape_time < self.config['antistuck']['escape_cooldown']:
            return

        print(f"INFO: Stuck! Trying escape stage {self.escape_stage + 1}...")
        self.control.release_all_keys()

        if self.escape_stage == 0:  # Stage 1: Simple strafe
            self.control.press('up')
            self.control.press('left' if random.random() < 0.5 else 'right')
            time.sleep(0.4)
        elif self.escape_stage == 1:  # Stage 2: Strafe with jump
            self.control.press('up')
            self.control.press('left' if error > 0 else 'right')
            self.control.press('jump')
            time.sleep(0.4)
        elif self.escape_stage == 2:  # Stage 3: Back up and turn
            self.control.press_for('down', 0.5)
            self.control.press_for('turn_right' if error < 0 else 'turn_left', 0.5)
        else:  # Stage 4: Full reposition
            self.control.press_for('down', self.config['antistuck']['escape_push_duration'] * 1.5)
            self.control.press_for('turn_right' if error < 0 else 'turn_left', 0.4)
            self.control.press('up')
            self.control.press('jump')
            time.sleep(self.config['antistuck']['escape_push_duration'])
            self.escape_stage = -1  # Reset after final attempt
        
        self.control.release_all_keys()
        self.last_escape_time = now
        self.stuck_frames = 0
        self.escape_stage += 1
        
    def handle_confirmation(self, is_target_visible: bool, target: Dict[str, Any], frame_shape: Tuple[int, int, int]) -> str:
        """
        Handles the logic for confirming a candy collection.

        Args:
            is_target_visible: Whether the target is still visible in the frame.
            target: The target being confirmed.
            frame_shape: The shape of the video frame.

        Returns:
            'collected', 'failed', or 'confirming'.
        """
        if self.confirmation_start_time == 0:
            self.confirmation_start_time = time.time()
            self.confirmation_failures = 0

        # If target is no longer visible, assume collected
        if not is_target_visible:
            print(f"[SUCCESS] Цель собрана!")
            self.confirmation_start_time = 0
            return 'collected'

        # If timeout is reached, assume collection failed
        if time.time() - self.confirmation_start_time > self.config['confirmation']['timeout']:
            print(f"[WARN] Не удалось подтвердить сбор цели, таймаут.")
            self.confirmation_start_time = 0
            self._add_to_memory(target, 'confirmation_fail', memory_type='long') # Add to long-term memory on timeout
            return 'failed'

        # If target is still visible, try to reposition
        self.confirmation_failures += 1
        print(f"[INFO] Цель все еще видна, попытка перемещения ({self.confirmation_failures})...")
        
        _, frame_w, _ = frame_shape
        center_x = frame_w / 2
        error_x = target['cx'] - center_x
        # Push forward aggressively during confirmation
        self.control.press_for('up', 0.1)
        self.perform_confirmation_reposition(self.confirmation_failures, error_x)

        return 'confirming'

    def perform_intelligent_search(self, current_time: float, search_intensity: float):
        """Controls the bot's behavior when no targets are visible."""
        self.control.release_all_keys()
        
        duration_multiplier = 1.0
        if search_intensity >= 3.0: duration_multiplier = 0.5
        elif search_intensity >= 2.0: duration_multiplier = 0.7

        pattern = self.config['search']['patterns'][self.current_search_pattern]

        if self.search_phase == 'SCANNING':
            scan_duration = self.config['search']['scan_duration'] * duration_multiplier
            if current_time - self.search_scan_start_time > scan_duration:
                self.search_phase = 'MOVING'
                self.search_move_start_time = current_time
            else:
                self._perform_scan_pattern(pattern)
        elif self.search_phase == 'MOVING':
            move_duration = self.config['search']['move_duration'] * duration_multiplier
            if current_time - self.search_move_start_time > move_duration:
                self.search_phase = 'SCANNING'
                self.search_scan_start_time = current_time
                self.search_turn_direction = random.choice(['turn_right', 'turn_left'])
                self.current_search_pattern = (self.current_search_pattern + 1) % len(self.config['search']['patterns'])
            else:
                self._perform_move_pattern(pattern, current_time)
    
    def _perform_scan_pattern(self, pattern: str):
        """Executes a specific scanning pattern."""
        if pattern == 'jump_scan' and random.random() < 0.3:
            self.control.press('jump')
        self.control.press(self.search_turn_direction)

    def _perform_move_pattern(self, pattern: str, current_time: float):
        """Executes a specific movement pattern during search."""
        self.control.press('up')
        if pattern == 'zigzag_move':
            strafe_key = 'left' if ((current_time - self.search_move_start_time) * 3) % 1.0 < 0.5 else 'right'
            self.control.press(strafe_key)
        elif pattern == 'spiral_search' and random.random() < 0.4:
            self.control.press('turn_left' if random.random() < 0.5 else 'turn_right')
        elif pattern in ['sprint_scan', 'jump_scan']:
            jump_chance = 0.2 if pattern == 'sprint_scan' else 0.4
            if random.random() < jump_chance: self.control.press('jump')

    def move_away_from_player(self, player: Dict[str, Any], frame_shape: Tuple[int, int, int]):
        """Executes a maneuver to flee from a nearby player."""
        frame_h, frame_w, _ = frame_shape
        center_x, center_y = frame_w / 2, frame_h / 2
        
        dx, dy = center_x - player['cx'], center_y - player['cy']
        
        self.control.release_all_keys()
        
        # Determine primary and secondary movement keys
        if abs(dx) > abs(dy):
            primary_move = 'right' if dx > 0 else 'left'
            secondary_move = 'up' if dy > 0 else 'down'
        else:
            primary_move = 'up' if dy > 0 else 'down'
            secondary_move = 'right' if dx > 0 else 'left'
            
        self.control.press(primary_move)
        if random.random() < 0.7:  # More varied movement
            self.control.press(secondary_move)
        
        # Turn away from the player
        turn_key = 'turn_right' if dx > 0 else 'turn_left'
        self.control.press(turn_key)
        
        if random.random() < 0.8: self.control.press('jump')
        
        # Increased duration for a more decisive escape
        time.sleep(0.4)
        self.control.release_all_keys()

    def perform_confirmation_reposition(self, failure_count: int, error_x: float):
        """Performs a small adjustment when a collection fails to confirm."""
        self.control.release_all_keys()
        pattern = failure_count % 3
        
        if pattern == 0:
            self.control.press_for('up', self.config['confirmation']['micro_push'])
        elif pattern == 1:
            strafe_key = 'left' if error_x > 0 else 'right'
            self.control.press('up')
            self.control.press(strafe_key)
            time.sleep(self.config['confirmation']['strafe_push'])
            self.control.release_all_keys()
        else:
            self.control.press_for('turn_left' if error_x > 0 else 'turn_right', self.config['confirmation']['turn_push'])
            
        time.sleep(0.02)

    def _calculate_turn_duration(self, error_x: float, last_frame_shape: Tuple[int, int, int]) -> float:
        """Calculates dynamic turn duration based on horizontal error."""
        if not self.config['dynamic_turn']['enabled'] or not last_frame_shape:
            return float(self.config['movement']['turn_base'])

        _, frame_w, _ = last_frame_shape
        half_width = float(frame_w) / 2.0
        if half_width <= 0: return float(self.config['movement']['turn_base'])
            
        normalized_error = min(1.0, abs(float(error_x)) / half_width)
        
        min_speed = float(self.config['dynamic_turn']['speed_min'])
        max_speed = float(self.config['dynamic_turn']['speed_max'])
        
        return min_speed + (max_speed - min_speed) * normalized_error

    def reset_short_term_memory(self):
        """Clears the short-term navigation memory."""
        self.short_term_memory = []
        print(f"[MEMORY] Кратковременная память очищена.")

    def _add_to_memory(self, target: Dict[str, Any], reason: str, memory_type: str = 'short'):
        """Adds a target's location to the specified memory."""
        if not target: return
        
        memory_entry = {
            'cx': target['cx'],
            'cy': target['cy'],
            'reason': reason,
            'timestamp': time.time()
        }

        if memory_type == 'short':
            self.short_term_memory.append(memory_entry)
            max_size = self.config['memory'].get('max_size', 20)
            if len(self.short_term_memory) > max_size:
                self.short_term_memory.pop(0)
        elif memory_type == 'long':
            self.long_term_memory['bad_spots'].append(memory_entry)
            self._save_map_cache() # Save immediately

    def filter_targets_from_memory(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out targets that are too close to a bad spot in memory."""
        if not self.short_term_memory and not self.long_term_memory['bad_spots']:
            return targets
            
        filtered_targets = []
        short_term_radius = self.config['memory'].get('memory_radius_px', 50)
        long_term_radius = self.config['memory'].get('long_term_radius_px', 75)
        
        for target in targets:
            is_bad = False
            # Check short-term memory
            for memory_spot in self.short_term_memory:
                dist_sq = (target['cx'] - memory_spot['cx'])**2 + (target['cy'] - memory_spot['cy'])**2
                if dist_sq < short_term_radius**2:
                    is_bad = True
                    break
            if is_bad: continue

            # Check long-term memory if not already flagged
            for memory_spot in self.long_term_memory['bad_spots']:
                dist_sq = (target['cx'] - memory_spot['cx'])**2 + (target['cy'] - memory_spot['cy'])**2
                if dist_sq < long_term_radius**2:
                    is_bad = True
                    break

            if not is_bad:
                filtered_targets.append(target)
                
        return filtered_targets
