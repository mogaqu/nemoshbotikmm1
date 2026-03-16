"""
Улучшенный бот для поиска мячиков в Murder Mystery 2
Версия 3.0 - Ultra Enhanced Ball Hunter Bot
"""
import sys
import os
import argparse
import time
import cv2
import numpy as np
import keyboard
import threading
from queue import Queue, Empty
from typing import List, Dict, Any, Optional, Tuple
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = BLUE = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roblox.screen import CaptureStream
from roblox.control import Control
from roblox.utils import FrameCounter
from components.config_manager import load_config
from components.detection_manager import DetectionManager
from components.navigation_system import NavigationSystem
from components.targeting_system import TargetingSystem
from components.ui_manager import UIManager

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.json')

# --- Global State ---
stop_requested = threading.Event()

def request_stop():
    """Requests the bot to stop gracefully."""
    if not stop_requested.is_set():
        print("\n[STOP] Команда остановки по клавише")
        stop_requested.set()

class EnhancedBallBot:
    """
    Orchestrates the bot's components to automate candy collection.
    """
    def __init__(self, config_path: str, args: argparse.Namespace, frame_queue: Queue):
        self.config = load_config(config_path)
        self._apply_args(args)
        self.frame_queue = frame_queue
        
        print(f"{Fore.CYAN}[INIT]{Style.RESET_ALL} Загрузка улучшенного бота: {Fore.YELLOW}{self.config['weights']['candy']}{Style.RESET_ALL}")
        if self.config['weights']['player']:
            print(f"{Fore.MAGENTA}[PREDATOR]{Style.RESET_ALL} Загрузка модели игроков: {Fore.YELLOW}{self.config['weights']['player']}{Style.RESET_ALL}")

        self.detection_manager = DetectionManager(self.config)
        self.targeting_system = TargetingSystem(self.config)
        self.ui_manager = UIManager(self.config, args.show, args.save_screenshots)
        
        self._initialize_components()
        self.navigation_system = NavigationSystem(self.config, self.control)
        
        self._initialize_state(args.no_adaptive)
        self._initialize_stats()
        self._initialize_ui_attributes()

        self.ui_manager.print_banner()
        self.ui_manager.print_startup_info(self)

    def _apply_args(self, args: argparse.Namespace):
        """Overrides config with command-line arguments."""
        if args.weights: self.config['weights']['candy'] = args.weights
        if args.player_weights: self.config['weights']['player'] = args.player_weights
        if args.conf: self.config['adaptive_confidence']['base_confidence'] = args.conf
        if args.player_conf: self.config['predator_mode']['player_conf_thres'] = args.player_conf
        if args.skip is not None: self.config['performance']['frame_skip'] = args.skip
        if args.player_skip is not None: self.config['performance']['player_detection_skip'] = args.player_skip
        if args.no_preprocess: self.config['preprocessing']['enabled'] = False

    def _initialize_components(self):
        """Initializes hardware-related components."""
        self.stream = CaptureStream("Roblox", saveInterval=0)
        self.control = Control()
        self.frame_counter = FrameCounter()
        self.frame_counter.fps = 60.0

    def _initialize_state(self, no_adaptive: bool):
        """Initializes the bot's dynamic state variables."""
        self.adaptive_mode: bool = not no_adaptive
        self.state: str = 'SEARCHING'
        self.current_target: Optional[Dict[str, Any]] = None
        self.target_for_confirmation: Optional[Dict[str, Any]] = None
        self.last_frame_shape: Optional[Tuple[int, int, int]] = None
        self.last_known_candies: Tuple[List[Dict[str, Any]], Any] = ([], None)
        self.confirmed_threats: List[Dict[str, Any]] = []
        self.threat_timestamps: Dict[int, float] = {}  # Track when each threat was last seen
        self.threat_id_counter: int = 0
        self.last_player_detection_time: float = 0.0
        self.no_detect_frames: int = 0
        self.search_intensity: float = 1.0
        self.last_candy_found_time: float = time.time()
        self.frame_counter_internal: int = 0
        self.conf_thres = self.config['adaptive_confidence']['base_confidence']
        self.last_frame_mean_intensity: float = -1.0
        self.fleeing_until: float = 0.0
        self.last_reset_time: float = 0.0

    def _initialize_stats(self):
        """Initializes statistics tracking."""
        self.stats = {
            'successful_collections': 0,
            'search_cycles': 0,
            'memory_revisits': 0,
            'start_time': time.time(),
            'pass_counts': {'base': 0, 'boost': 0, 'heavy': 0},
            'players_detected': 0,
            'confirmation_failures': 0,
            'total_detections': 0,
            'cache_retrievals': 0,
        }
        self.conf_sum: float = 0.0

    def _initialize_ui_attributes(self):
        """Initializes attributes needed by the UI manager."""
        self.min_distance = self.config['movement']['min_distance']
        self.frame_skip = self.config['performance']['frame_skip']
        self.use_fp16 = self.config['predator_mode']['use_fp16']
        self.manual_stop_key = self.config['general']['manual_stop_key']
        self.last_detection_pass = 'base'

    def run(self):
        """The main execution loop of the bot. This runs in a worker thread."""
        keyboard.add_hotkey(self.config['general']['manual_stop_key'], request_stop)

        try:
            for img, _ in self.stream:
                if stop_requested.is_set(): break
                
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                current_time = time.time()
                self.last_frame_shape = img.shape
                self.frame_counter_internal += 1

                # Player detection and fleeing logic now happens BEFORE target processing
                is_currently_fleeing = self._handle_player_threats()

                # If we are fleeing, we don't process targets or the normal FSM
                if is_currently_fleeing:
                    if self.ui_manager.show_window:
                        annotated_frame = self.ui_manager.prepare_display_frame(img, None, self)
                        self.frame_queue.put(annotated_frame)
                    self.frame_counter.log()
                    continue

                candy_targets, candy_results = self._update_targets(img, current_time)
                best_target = candy_targets[0] if candy_targets else None
                
                self._update_fsm(best_target, candy_targets, current_time)

                self._check_for_round_reset(img)
                self._update_adaptive_confidence(candy_targets)

                # Instead of displaying, put the annotated frame into the queue
                if self.ui_manager.show_window:
                    annotated_frame = self.ui_manager.prepare_display_frame(img, candy_results, self)
                    self.frame_queue.put(annotated_frame)
                
                self.frame_counter.log()
        finally:
            self._shutdown()

    def _update_targets(self, img: np.ndarray, current_time: float) -> Tuple[List[Dict[str, Any]], Any]:
        """Detects objects and returns prioritized targets."""
        det_img = self.detection_manager.preprocess_frame(img)
        
        # Player detection
        player_skip = self.config['performance']['player_detection_skip']
        if self.detection_manager.player_model and (player_skip == 0 or self.frame_counter_internal % player_skip == 0):
            player_conf = self.targeting_system.get_dynamic_player_conf(current_time, self.last_player_detection_time)
            raw_players, _ = self.detection_manager.run_player_detection(det_img, player_conf)
            if raw_players: self.last_player_detection_time = current_time
            current_players = self.targeting_system.filter_ego(raw_players, det_img.shape)
            self.confirmed_threats = self.targeting_system.confirm_threats(current_players, self.threat_timestamps, current_time)
            self._update_threat_timestamps(current_players, current_time)
            self._cleanup_old_threats(current_time)
            if self.confirmed_threats: self.stats['players_detected'] += len(self.confirmed_threats)

        # Candy detection
        frame_skip = self.config['performance']['frame_skip']
        if frame_skip == 0 or self.frame_counter_internal % frame_skip == 0:
            targets, results, pass_name = self.detection_manager.run_candy_detection(det_img, self.conf_thres, self.no_detect_frames)
            self.last_known_candies = (targets, results)
            self.last_detection_pass = pass_name
            self.stats['pass_counts'][pass_name] += 1
            if targets:
                self.stats['total_detections'] += len(targets)
                # Accumulate confidence sum for statistics
                for target in targets:
                    self.conf_sum += target.get('conf', 0.0)
            if pass_name == 'heavy': self.no_detect_frames = 0
        else:
            targets, results = self.last_known_candies

        prioritized_targets = self.targeting_system.find_prioritized_targets(targets, img.shape, self.confirmed_threats)
        # Filter out targets that are in bad spots from memory
        return self.navigation_system.filter_targets_from_memory(prioritized_targets), results

    def _handle_player_threats(self) -> bool:
        """
        Handles escape logic if a player is too close.
        Returns True if the bot is in a fleeing state, False otherwise.
        """
        # First, check if we are on a flee cooldown. If so, do nothing but report we are "busy" fleeing.
        if time.time() < self.fleeing_until:
            return True

        # If cooldown is over, check for new threats
        if not self.confirmed_threats or not self.last_frame_shape:
            return False

        closest, dist = self.targeting_system.get_closest_threat(self.confirmed_threats, self.last_frame_shape)
        if closest and dist < self.config['predator_mode']['player_escape_distance']:
            print(f"{Fore.RED}[ESCAPE]{Style.RESET_ALL} Игрок слишком близко! Убегаю...")
            self.control.release_all_keys()
            self.state = 'SEARCHING'
            self.current_target = None  # Explicitly clear target
            self.navigation_system.move_away_from_player(closest, self.last_frame_shape)
            # Set the cooldown AFTER the maneuver
            self.fleeing_until = time.time() + 2.5
            return True # We initiated a flee, so we are busy

        return False

    def _update_fsm(self, best_target, candy_targets, current_time):
        """Manages the bot's finite state machine."""
        # The fleeing check is now outside the FSM, so we remove it from here.
        if self.state == 'HUNTING':
            if best_target:
                self.current_target = best_target
            else: self.state = 'SEARCHING'; return
            
            status, err_x = self.navigation_system.move_to_target(self.current_target, self.last_frame_shape, self.frame_counter.fps)
            if status == 'at_target': self.state = 'CONFIRMING'
            elif status == 'stuck':
                self.navigation_system.perform_escape_maneuver(err_x)
                self.targeting_system.forget_target(self.current_target, self.last_frame_shape)
                self.current_target = None
                self.state = 'SEARCHING'

        elif self.state == 'CONFIRMING':
            is_visible = self.targeting_system.is_target_still_visible(self.current_target, candy_targets)
            result = self.navigation_system.handle_confirmation(is_visible, self.current_target, self.last_frame_shape)
            if result == 'collected':
                self.stats['successful_collections'] += 1
                self.targeting_system.forget_target(self.current_target, self.last_frame_shape)
                self.current_target = None
                self.state = 'SEARCHING'
            elif result == 'failed':
                # The navigation system already added it to memory
                self.current_target = None
                self.state = 'SEARCHING'

        elif self.state == 'SEARCHING':
            if best_target:
                self.current_target = best_target
                self.state = 'HUNTING'
                self.last_candy_found_time = current_time
            else:
                self.navigation_system.perform_intelligent_search(current_time, self.search_intensity)

    def _check_for_round_reset(self, frame: np.ndarray):
        """Checks for a flash or blackout indicating a round change."""
        # Use a small, resized version for performance
        resized_frame = cv2.resize(frame, (32, 32))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        current_intensity = np.mean(gray_frame)

        # Initialize on the first frame
        if self.last_frame_mean_intensity < 0:
            self.last_frame_mean_intensity = current_intensity
            return

        flash_threshold = self.config['general'].get('flash_detection_threshold', 80.0)
        blackout_threshold = self.config['general'].get('blackout_detection_threshold', 15.0)
        
        is_flash = abs(current_intensity - self.last_frame_mean_intensity) > flash_threshold
        is_blackout = current_intensity < blackout_threshold and self.last_frame_mean_intensity < blackout_threshold

        # Cooldown to prevent multiple triggers
        now = time.time()
        if (is_flash or is_blackout) and (now - self.last_reset_time > 5.0):
            print(f"{Fore.MAGENTA}[SYSTEM]{Style.RESET_ALL} Обнаружена смена раунда! Сброс кратковременной памяти.")
            self.navigation_system.reset_short_term_memory()
            # Clear all threats when round changes
            self.confirmed_threats = []
            self.threat_timestamps = {}
            # Also reset search state to avoid weird initial turns
            self.navigation_system.search_phase = 'SCANNING'
            self.navigation_system.search_scan_start_time = time.time()
            self.last_reset_time = now

        self.last_frame_mean_intensity = current_intensity

    def _update_threat_timestamps(self, current_players: List[Dict[str, Any]], current_time: float):
        """Updates timestamps for currently detected players."""
        if not current_players:
            return
        
        for player in current_players:
            # Create a simple ID based on position (rounded to avoid small movements)
            player_id = int(player['cx'] / 20) * 1000 + int(player['cy'] / 20)
            self.threat_timestamps[player_id] = current_time

    def _cleanup_old_threats(self, current_time: float):
        """Removes threats that haven't been seen for too long."""
        threat_timeout = self.config['predator_mode'].get('threat_timeout', 1.5)  # 1.5 seconds
        
        # Remove threats that haven't been seen recently
        threats_to_remove = []
        for i, threat in enumerate(self.confirmed_threats):
            threat_id = int(threat['cx'] / 20) * 1000 + int(threat['cy'] / 20)
            last_seen = self.threat_timestamps.get(threat_id, 0)
            
            if current_time - last_seen > threat_timeout:
                threats_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(threats_to_remove):
            self.confirmed_threats.pop(i)

    def _update_adaptive_confidence(self, candy_targets):
        """Adjusts confidence based on search performance."""
        if not self.adaptive_mode: return

        time_since_last = time.time() - self.last_candy_found_time
        search_cfg = self.config['search']
        if time_since_last > search_cfg['very_aggressive_threshold']: self.search_intensity = 3.0
        elif time_since_last > search_cfg['aggressive_threshold']: self.search_intensity = 2.0
        else: self.search_intensity = 1.0
        
        if not candy_targets: self.no_detect_frames += 1
        else: self.no_detect_frames = 0
            
        self.conf_thres = self.targeting_system.get_adaptive_confidence(
            self.config['adaptive_confidence']['base_confidence'], self.no_detect_frames, self.search_intensity
        )

    def _shutdown(self):
        """Gracefully shuts down the bot."""
        print(f"\n{Fore.CYAN}[EXIT]{Style.RESET_ALL} Завершение работы...")
        self.control.release_all_keys()
        self.ui_manager.print_stats(self.stats, self.frame_counter, self.search_intensity, self.conf_sum)
        keyboard.remove_all_hotkeys()
        print(f"{Fore.GREEN}[EXIT]{Style.RESET_ALL} Бот остановлен")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Candy Hunter Bot v3.0')
    parser.add_argument('--weights', type=str, help='Путь к весам модели для конфет')
    parser.add_argument('--player-weights', type=str, help='Путь к весам модели для игроков')
    parser.add_argument('--conf', type=float, help='Базовый порог уверенности')
    parser.add_argument('--no-adaptive', action='store_true', help='Отключить адаптивную систему')
    parser.add_argument('--show', action='store_true', help='Показать окно с аннотированным изображением')
    parser.add_argument('--save-screenshots', action='store_true', help='Сохранять скриншоты')
    parser.add_argument('--skip', type=int, help='Пропуск кадров для детекции конфет')
    parser.add_argument('--player-skip', type=int, help='Пропуск кадров для детекции игроков')
    parser.add_argument('--player-conf', type=float, help='Порог уверенности для игроков')
    parser.add_argument('--no-preprocess', action='store_true', help='Отключить предобработку изображений')
    args = parser.parse_args()

    frame_queue = Queue(maxsize=2)
    bot = EnhancedBallBot(config_path=DEFAULT_CONFIG_PATH, args=args, frame_queue=frame_queue)
    
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()

    if args.show:
        while not stop_requested.is_set():
            try:
                frame = frame_queue.get(timeout=1)
                cv2.imshow('🎯 Ultra Enhanced Bot - View', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    request_stop()
            except Empty:
                if not bot_thread.is_alive():
                    break
        cv2.destroyAllWindows()
    
    bot_thread.join()

if __name__ == '__main__':
    main()
