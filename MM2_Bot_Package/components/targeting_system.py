import numpy as np
import time
from collections import deque
from typing import Dict, Any, List, Optional, Deque, Tuple

class TargetingSystem:
    """Manages target selection, prioritization, and memory."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TargetingSystem.

        Args:
            config: The main configuration dictionary for the bot.
        """
        self.config: Dict[str, Any] = config
        self.target_memory: Deque[Dict[str, Any]] = deque(maxlen=config['memory']['deque_maxlen'])
        self.hot_zones: Deque[Dict[str, Any]] = deque(maxlen=5)
        
        # Parameters from config for quick access
        self.y_axis_priority_weight: float = config['targeting']['y_axis_priority_weight']
        self.risk_factor_weight: float = config['predator_mode']['risk_factor_weight']
        self.risk_distance_threshold: int = config['predator_mode']['risk_distance_threshold']
        self.memory_retention: float = config['memory']['retention']
        self.memory_min_conf: float = config['memory']['min_conf']
        self.memory_revisit_cooldown: float = config['memory']['revisit_cooldown']
        
        # Get target classes from config (supports both ball and coins)
        target_classes = config.get('target_classes', {'ball': 0})
        self.target_class_ids: set = set(target_classes.values())
        
        # State variables
        self.last_revisit_time: float = 0.0
        self.last_detection_time: float = 0.0
        self.last_frame_targets: List[Dict[str, Any]] = []

    def find_prioritized_targets(self, targets: List[Dict[str, Any]], frame_shape: Tuple[int, int, int], player_targets: List[Dict[str, Any]], max_targets: int = 3) -> List[Dict[str, Any]]:
        """
        Finds and prioritizes targets considering players, stability, and other factors.

        Args:
            targets: A list of raw detected targets from the current frame.
            frame_shape: The shape of the current video frame.
            player_targets: A list of detected player targets.
            max_targets: The maximum number of prioritized targets to return.

        Returns:
            A sorted list of the highest-priority targets.
        """
        if not targets:
            self.last_frame_targets = []
            return []

        frame_h, frame_w, _ = frame_shape
        center_x, center_y = frame_w / 2, frame_h / 2
        frame_time = time.time()
        
        current_frame_targets: List[Dict[str, Any]] = []
        for t in targets:
            # Filter by target classes from config (supports ball=0 and coins=1)
            if t['cls'] not in self.target_class_ids: continue

            stability = 1
            for prev_t in self.last_frame_targets:
                dist = np.hypot(t['cx'] - prev_t['cx'], t['cy'] - prev_t['cy'])
                if dist < 30:
                    stability = prev_t.get('stability', 1) + 1
                    break
            
            t['stability'] = stability
            current_frame_targets.append(t)

        prioritized_targets: List[Dict[str, Any]] = []
        for t in current_frame_targets:
            cx, cy = t['cx'], t['cy']
            
            distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            was_memory = self._maybe_refresh_memory(t, frame_time, frame_shape)

            priority_score = self._calculate_priority_score(t, distance, frame_shape, player_targets)
            
            prioritized_targets.append({
                **t,
                'distance': distance,
                'priority_score': priority_score,
                'memory_boost': was_memory
            })
        
        prioritized_targets.sort(key=lambda t: t['priority_score'], reverse=True)
        self.last_frame_targets = current_frame_targets
        return prioritized_targets[:max_targets]

    def _calculate_priority_score(self, target: Dict[str, Any], distance: float, frame_shape: Tuple[int, int, int], player_targets: List[Dict[str, Any]]) -> float:
        """Calculates a priority score for a single target."""
        cx, cy, conf, stability = target['cx'], target['cy'], target['conf'], target['stability']
        frame_h, frame_w, _ = frame_shape
        center_x, center_y = frame_w / 2, frame_h / 2
        
        priority = conf * 100
        priority += max(0, 50 - abs(cx - center_x) / 8)
        priority += max(0, 80 - abs(cy - center_y) / 4) * self.y_axis_priority_weight
        priority += max(0, 130 - distance / 4)
        priority += self._get_hot_zone_bonus(cx, cy, frame_shape)
        priority += (stability - 1) * 25
        
        risk = self._calculate_risk_factor(cx, cy, player_targets)
        priority -= risk * self.risk_factor_weight

        return priority

    def _calculate_risk_factor(self, candy_x: float, candy_y: float, player_targets: List[Dict[str, Any]]) -> float:
        """Calculates the risk factor for a candy based on player proximity."""
        if not player_targets:
            return 0.0

        total_risk = 0.0
        for player in player_targets:
            dist_to_player = np.hypot(candy_x - player['cx'], candy_y - player['cy'])
            if dist_to_player < self.risk_distance_threshold:
                risk = (1 - (dist_to_player / self.risk_distance_threshold))
                total_risk += risk

        return total_risk

    def _clean_target_memory(self, current_time: float):
        """Removes old and low-confidence targets from memory."""
        if not self.target_memory:
            return
        filtered = [entry for entry in self.target_memory
                    if current_time - entry['time'] <= self.memory_retention and entry['conf'] >= self.memory_min_conf]
        if len(filtered) < len(self.target_memory):
            self.target_memory.clear()
            self.target_memory.extend(filtered)

    def _maybe_refresh_memory(self, entry: Dict[str, Any], frame_time: float, frame_shape: Tuple[int, int, int]) -> bool:
        """Refreshes an existing target in memory or returns False if no match is found."""
        frame_h, frame_w, _ = frame_shape
        if frame_w == 0 or frame_h == 0:
            return False

        self._clean_target_memory(frame_time)

        cx_norm, cy_norm = entry['cx'] / frame_w, entry['cy'] / frame_h

        for memory in self.target_memory:
            dist = np.hypot((cx_norm - memory['cx_norm']) * frame_w, (cy_norm - memory['cy_norm']) * frame_h)
            if dist < 70:
                memory.update({
                    'cx_norm': cx_norm,
                    'cy_norm': cy_norm,
                    'conf': max(memory['conf'], float(entry['conf'])),
                    'priority': max(memory.get('priority', 0.0), float(entry.get('priority_score', entry['conf'] * 100.0))),
                    'time': frame_time
                })
                return True
        return False

    def update_target_memory(self, target: Dict[str, Any], frame_time: float, frame_shape: Tuple[int, int, int]):
        """Adds or updates a target in the memory."""
        if frame_shape is None: return
        
        if not self._maybe_refresh_memory(target, frame_time, frame_shape):
            self._clean_target_memory(frame_time)
            frame_h, frame_w, _ = frame_shape
            if frame_w == 0 or frame_h == 0: return
            
            self.target_memory.append({
                'cx_norm': float(target['cx'] / frame_w),
                'cy_norm': float(target['cy'] / frame_h),
                'conf': float(target['conf']),
                'priority': float(target.get('priority_score', target['conf'] * 100.0)),
                'time': frame_time
            })
        self.last_detection_time = frame_time

    def forget_target(self, target: Dict[str, Any], frame_shape: Tuple[int, int, int]):
        """Removes a specific target from memory."""
        if frame_shape is None or not self.target_memory: return
        frame_h, frame_w, _ = frame_shape
        if frame_w == 0 or frame_h == 0: return

        self._clean_target_memory(time.time())

        target_cx_norm = target['cx'] / frame_w
        target_cy_norm = target['cy'] / frame_h

        remaining = [entry for entry in self.target_memory
                     if np.hypot(target_cx_norm - entry['cx_norm'], target_cy_norm - entry['cy_norm']) > 0.03]
        if len(remaining) < len(self.target_memory):
            self.target_memory.clear()
            self.target_memory.extend(remaining)

    def select_memory_target(self) -> Optional[Dict[str, Any]]:
        """Selects the best target from memory to revisit."""
        current_time = time.time()
        if current_time - self.last_revisit_time < self.memory_revisit_cooldown or \
           current_time - self.last_detection_time < 0.4:
            return None

        self._clean_target_memory(current_time)
        if not self.target_memory:
            return None

        return max(self.target_memory, key=lambda entry: (entry.get('priority', 0.0), entry['conf']))

    def _get_hot_zone_bonus(self, cx: float, cy: float, frame_shape: Tuple[int, int, int]) -> float:
        """Calculates a priority bonus for proximity to 'hot zones'."""
        if not self.hot_zones: return 0
        frame_h, frame_w, _ = frame_shape
        max_bonus = 25

        total_bonus = sum(
            max_bonus * (1.0 - np.sqrt((cx - z['x'] * frame_w)**2 + (cy - z['y'] * frame_h)**2) / 150) * z['confidence']
            for z in self.hot_zones
            if time.time() - z['time'] <= 60 and np.sqrt((cx - z['x'] * frame_w)**2 + (cy - z['y'] * frame_h)**2) < 150
        )
        return min(total_bonus, max_bonus * 1.5)

    def filter_ego(self, player_targets: List[Dict[str, Any]], frame_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Filters out the player's own character detection."""
        if not player_targets:
            return []
        
        frame_h, frame_w, _ = frame_shape
        x1_ego, y1_ego, x2_ego, y2_ego = self.config['predator_mode']['ego_filter_zone']
        ego_zone = (x1_ego * frame_w, y1_ego * frame_h, x2_ego * frame_w, y2_ego * frame_h)

        return [p for p in player_targets if not (ego_zone[0] < p['cx'] < ego_zone[2] and ego_zone[1] < p['cy'] < ego_zone[3])]

    def add_hot_zone(self, target: Dict[str, Any], frame_shape: Tuple[int, int, int]):
        """Adds a new hot zone based on a collected target's location."""
        frame_h, frame_w, _ = frame_shape
        self.hot_zones.append({
            'x': target['cx'] / frame_w,
            'y': target['cy'] / frame_h,
            'time': time.time(),
            'confidence': target['conf']
        })

    def get_adaptive_confidence(self, base_conf: float, no_detect_frames: int, search_intensity: float) -> float:
        """
        Calculates the adaptive confidence threshold.

        Args:
            base_conf: The base confidence threshold from the config.
            no_detect_frames: The number of consecutive frames without a detection.
            search_intensity: The current search intensity level.

        Returns:
            The adjusted confidence threshold.
        """
        conf_range = self.config['adaptive_confidence']['confidence_range']
        max_no_detect = self.config['adaptive_confidence']['max_no_detect_frames']
        
        reduction_factor = min(1.0, no_detect_frames / max_no_detect)
        new_conf = base_conf - (conf_range * reduction_factor)
        
        if search_intensity >= 3.0:
            new_conf -= self.config['adaptive_confidence']['intensity_modifier_heavy']
        elif search_intensity >= 2.0:
            new_conf -= self.config['adaptive_confidence']['intensity_modifier_aggressive']
            
        return max(self.config['adaptive_confidence']['min_confidence'], new_conf)

    def get_dynamic_player_conf(self, current_time: float, last_detection_time: float) -> float:
        """
        Adjusts player detection confidence based on how recently players were seen.

        Args:
            current_time: The current timestamp.
            last_detection_time: Timestamp of the last player detection.

        Returns:
            The adjusted player confidence threshold.
        """
        base_conf = self.config['predator_mode']['player_conf_thres']
        time_since = current_time - last_detection_time
        if time_since < self.config['predator_mode']['confidence_boost_duration']:
            return base_conf + self.config['predator_mode']['confidence_boost_amount']
        return base_conf

    def confirm_threats(self, current_players: List[Dict[str, Any]], threat_timestamps: Dict[int, float], current_time: float) -> List[Dict[str, Any]]:
        """
        Confirms threats by checking for consistent detection across frames.
        Only returns threats that have been seen recently.

        Args:
            current_players: List of players detected in the current frame.
            threat_timestamps: Dictionary tracking when each threat was last seen.
            current_time: Current timestamp.

        Returns:
            A list of confirmed player threats.
        """
        if not current_players:
            return []
        
        threat_timeout = self.config['predator_mode'].get('threat_timeout', 1.5)
        confirmed = []
        
        for player in current_players:
            # Create a simple ID based on position
            player_id = int(player['cx'] / 20) * 1000 + int(player['cy'] / 20)
            last_seen = threat_timestamps.get(player_id, current_time)
            
            # Confirm if player was seen recently (within timeout) or is new
            time_since_seen = current_time - last_seen
            if time_since_seen <= threat_timeout:
                confirmed.append(player)
        
        return confirmed

    def get_closest_threat(self, threats: List[Dict[str, Any]], frame_shape: Tuple[int, int, int]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Finds the closest player threat to the center of the screen.

        Args:
            threats: A list of confirmed player threats.
            frame_shape: The shape of the video frame.

        Returns:
            A tuple containing the closest threat and its distance, or (None, infinity).
        """
        if not threats:
            return None, float('inf')
        
        frame_h, frame_w, _ = frame_shape
        center_x, center_y = frame_w / 2, frame_h / 2

        closest_threat = min(
            threats,
            key=lambda t: np.hypot(t['cx'] - center_x, t['cy'] - center_y)
        )
        distance = np.hypot(closest_threat['cx'] - center_x, closest_threat['cy'] - center_y)
        return closest_threat, distance

    def is_target_still_visible(self, target: Dict[str, Any], current_targets: List[Dict[str, Any]]) -> bool:
        """
        Checks if the specified target is still visible in the current list of detections.

        Args:
            target: The target to check for.
            current_targets: The list of targets detected in the current frame.

        Returns:
            True if the target is still visible, False otherwise.
        """
        if not target or not current_targets:
            return False
        
        visibility_threshold = self.config['confirmation']['visibility_threshold']
        for t in current_targets:
            dist = np.hypot(t['cx'] - target['cx'], t['cy'] - target['cy'])
            if dist < visibility_threshold:
                return True
        return False
