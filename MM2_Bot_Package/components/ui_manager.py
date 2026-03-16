import cv2
import numpy as np
import time
from colorama import Fore, Style
from typing import Dict, Any, List, Optional, Tuple

class UIManager:
    """Manages all UI-related tasks, including the OpenCV window, HUD, and statistics display."""

    def __init__(self, config: Dict[str, Any], show_window: bool = False, save_screenshots: bool = False):
        """
        Initializes the UIManager.

        Args:
            config: The main configuration dictionary for the bot.
            show_window: Whether to display the bot's view in an OpenCV window.
            save_screenshots: Whether to enable saving screenshots.
        """
        self.config: Dict[str, Any] = config
        self.show_window: bool = show_window
        self.save_screenshots: bool = save_screenshots
        self.window_failed: bool = False
        self.screenshot_counter: int = 0

        if self.show_window:
            self._test_opencv_gui()

    def _test_opencv_gui(self):
        """Tests if the OpenCV GUI is available and functional."""
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('OpenCV Test', test_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} OpenCV GUI not available: {e}. Falling back to saving screenshots.")
            self.show_window = False
            self.save_screenshots = True
            self.window_failed = True

    def print_startup_info(self, bot_state: Any):
        """
        Prints the startup information panel.

        Args:
            bot_state: The main bot instance to source data from.
        """
        print(f"{Fore.CYAN}╔═══════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}✓ Улучшенный бот готов!{Style.RESET_ALL}                                    {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╠═══════════════════════════════════════════════════════════════╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}Confidence:{Style.RESET_ALL} {bot_state.conf_thres:.3f} | {Fore.YELLOW}Adaptive:{Style.RESET_ALL} {'ON' if bot_state.adaptive_mode else 'OFF'}{' ' * 30}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}Min Distance:{Style.RESET_ALL} {bot_state.min_distance}px | {Fore.YELLOW}Predict Size:{Style.RESET_ALL} Dynamic{' ' * 19}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}Frame Skip:{Style.RESET_ALL} {bot_state.frame_skip}{' ' * 48}{Fore.CYAN}║{Style.RESET_ALL}")
        if bot_state.detection_manager.player_model:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.MAGENTA}🔪 Predator Mode:{Style.RESET_ALL} {Fore.RED}ON{Style.RESET_ALL}{' ' * 38}{Fore.CYAN}║{Style.RESET_ALL}")
            if bot_state.use_fp16:
                print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}⚡ FP16 Mode:{Style.RESET_ALL} Enabled{' ' * 43}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}[START]{Style.RESET_ALL} Бот запущен! {Fore.YELLOW}Ctrl+C{Style.RESET_ALL} или клавиша {Fore.YELLOW}'{bot_state.manual_stop_key}'{Style.RESET_ALL} для остановки\n")

    def print_banner(self):
        """Prints a cool ASCII art banner to the console."""
        banner = f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════════════╗
║{Style.RESET_ALL} {Fore.YELLOW}╔═══╗╔═╗ ╔╗╔═╗╔╗╔╗╔═╗╔═╗╔╗╔╗╔═╗╔═╗╔╗╔╗{Style.RESET_ALL} {Fore.CYAN}   ║
║{Style.RESET_ALL} {Fore.YELLOW}║╔═╗║║╔╝ ║║║╔╝║║║║║║║║╔╝║║║║║║║║║╔╝║║║║║║{Style.RESET_ALL} {Fore.CYAN}   ║
║{Style.RESET_ALL} {Fore.YELLOW}║╚═╝║║║  ║║║║ ║╚╝║║║║ ║║║║║║║║║║ ║║║║║║{Style.RESET_ALL} {Fore.CYAN}   ║
║{Style.RESET_ALL} {Fore.YELLOW}║╔═╗║║║ ╔╣║║║ ║╔╗║║║║ ║║║║║║║║║║ ║║║║║║{Style.RESET_ALL} {Fore.CYAN}   ║
║{Style.RESET_ALL} {Fore.YELLOW}║║ ║║║╚═╝║║║╚╗║║║║║║╚╗║║║║║║║║║╚╗║║║║║║{Style.RESET_ALL} {Fore.CYAN}   ║
║{Style.RESET_ALL} {Fore.YELLOW}╚╝ ╚╝╚═══╝╚╝╚═╝╚╝╚╝╚═╝╚═╝╚╝╚╝╚═╝╚═╝╚╝╚╝{Style.RESET_ALL} {Fore.CYAN}   ║
║{Style.RESET_ALL} {Fore.GREEN}Version 3.0 - Ultra Enhanced Edition{Style.RESET_ALL} {Fore.CYAN}                ║
╚═══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)

    def print_stats(self, stats: Dict[str, Any], frame_counter: Any, search_intensity: float, conf_sum: float):
        """
        Prints a formatted table of runtime statistics.

        Args:
            stats: Dictionary containing bot statistics.
            frame_counter: The FrameCounter instance.
            search_intensity: The current search intensity level.
            conf_sum: The cumulative sum of detection confidences.
        """
        runtime = time.time() - stats['start_time']
        
        avg_confidence = (conf_sum / stats['total_detections']) if stats['total_detections'] > 0 else 0
        eff_percent = (stats['successful_collections'] / max(1, stats['total_detections']) * 100)
        search_efficiency = (stats['total_detections'] / max(1, stats['search_cycles']))
        
        runtime_str = f"{int(runtime//60)}m {int(runtime%60)}s" if runtime > 60 else f"{runtime:.1f}s"
        eff_color = Fore.GREEN if eff_percent > 70 else Fore.YELLOW if eff_percent > 50 else Fore.RED
        
        intensity_indicator = "Normal"
        if search_intensity >= 3.0: intensity_indicator = f"{Fore.RED}Super-Aggressive{Style.RESET_ALL}"
        elif search_intensity >= 2.0: intensity_indicator = f"{Fore.YELLOW}Aggressive{Style.RESET_ALL}"
            
        fps_color = Fore.GREEN if frame_counter.fps > 30 else Fore.YELLOW if frame_counter.fps > 15 else Fore.RED
        
        passes_str = f"base:{stats['pass_counts']['base']}, boost:{stats['pass_counts']['boost']}, heavy:{stats['pass_counts']['heavy']}"

        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}📊 BOT STATISTICS{Style.RESET_ALL} {' ' * 42}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╠═══════════════════════════════════════════════════════════════╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}⏱ Время работы:{Style.RESET_ALL} {runtime_str:<45}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}🎯 Детекций:{Style.RESET_ALL} {stats['total_detections']:<49}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}💰 Собрано:{Style.RESET_ALL} {stats['successful_collections']:<49}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}📈 Эффективность:{Style.RESET_ALL} {eff_color}{eff_percent:.1f}%{Style.RESET_ALL}{' ' * (43-len(f'{eff_percent:.1f}%'))}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}🔄 Циклов поиска:{Style.RESET_ALL} {stats['search_cycles']:<45}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}🔍 Эффективность поиска:{Style.RESET_ALL} {search_efficiency:.1f} D/C{' ' * (38-len(f'{search_efficiency:.1f} D/C'))}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}⚡ Режим поиска:{Style.RESET_ALL} {intensity_indicator:<40}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}🎲 Ср. уверенность:{Style.RESET_ALL} {avg_confidence:.3f}{' ' * (43-len(f'{avg_confidence:.3f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}🔍 Детекция:{Style.RESET_ALL} {passes_str:<42}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}🧠 Возвратов к памяти:{Style.RESET_ALL} {stats['memory_revisits']:<38}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}💾 Кэш переключений:{Style.RESET_ALL} {stats['cache_retrievals']:<38}{Fore.CYAN}║{Style.RESET_ALL}")
        if stats.get('players_detected', 0) > 0:
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.RED}🔪 Игроков обнаружено:{Style.RESET_ALL} {stats['players_detected']:<37}{Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.YELLOW}⚠ Провалов подтверждения:{Style.RESET_ALL} {stats['confirmation_failures']:<34}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}⚡ Текущий FPS:{Style.RESET_ALL} {fps_color}{frame_counter.fps:.1f}{Style.RESET_ALL}{' ' * (43-len(f'{frame_counter.fps:.1f}'))}{Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")

    def prepare_display_frame(self, img: np.ndarray, results: Any, bot_state: Any) -> np.ndarray:
        """
        Renders annotations and HUD on the frame and returns it.
        This version does not display the image itself.

        Args:
            img: The original BGR image frame.
            results: The raw YOLO detection results for plotting.
            bot_state: The main bot instance to source data from for the HUD.

        Returns:
            The annotated frame as a numpy array.
        """
        try:
            annotated = results[0].plot() if results and hasattr(results[0], 'plot') else img.copy()
            h, w, _ = annotated.shape

            annotated = self._draw_gradient_overlay(annotated, alpha=0.15)
            self._draw_crosshair(annotated, w, h)
            self._draw_player_warnings(annotated, bot_state.confirmed_threats)
            self._draw_current_target(annotated, bot_state.current_target, img.shape)
            self._draw_hud(annotated, bot_state)
            self._draw_fps_bar(annotated, bot_state.frame_counter.fps, w)
            self._draw_detected_targets(annotated, bot_state.last_known_candies[0], img.shape)

            if self.save_screenshots:
                # Save screenshot if requested via hotkey 's'
                # This check might need to be adapted depending on how hotkeys are handled now
                pass

            return annotated

        except Exception as e:
            print(f'{Fore.RED}[WARN] Display error: {e}{Style.RESET_ALL}')
            # Return original image on error to prevent crash
            return img

    def display_frame(self, img: np.ndarray, results: Any, bot_state: Any) -> Optional[str]:
        """
        Renders annotations and HUD on the frame and displays it.

        Args:
            img: The original BGR image frame.
            results: The raw YOLO detection results for plotting.
            bot_state: The main bot instance to source data from for the HUD.

        Returns:
            'stop' if the user requested to quit, otherwise None.
        """
        try:
            annotated = self.prepare_display_frame(img, results, bot_state)
            
            if self.save_screenshots:
                # Placeholder for screenshot logic
                pass
            
            if self.show_window:
                cv2.imshow('🎯 Ultra Enhanced Bot - View', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return 'stop'
                elif key == ord('s'):
                    filename = f"debug_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated)
                    print(f"{Fore.GREEN}[DEBUG] Screenshot saved: {filename}{Style.RESET_ALL}")

        except Exception as e:
            print(f'{Fore.RED}[WARN] Display error: {e}{Style.RESET_ALL}')
            if not self.window_failed:
                self.show_window = False
                self.save_screenshots = True
                self.window_failed = True
        return None

    def _draw_gradient_overlay(self, img: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Draws a semi-transparent gradient on the top of the image."""
        overlay = img.copy()
        h, w, _ = img.shape
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            # The alpha value decreases as we go down the screen
            overlay_alpha = alpha * (1 - (y / h))
            cv2.line(gradient, (0, y), (w, y), (0, 0, 0), 1)
        
        return cv2.addWeighted(gradient, alpha, overlay, 1 - alpha, 0)

    def _draw_beautiful_text(self, img: np.ndarray, text: str, pos: Tuple[int, int], font_scale: float = 0.7, thickness: int = 2, color: Tuple[int, int, int] = (0, 255, 255), bg_alpha: float = 0.5):
        """Draws text with a semi-transparent background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = pos
        
        sub_img = img[y-text_h-5:y+baseline+5, x-5:x+text_w+5]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        
        res = cv2.addWeighted(sub_img, 1 - bg_alpha, black_rect, bg_alpha, 0)
        img[y-text_h-5:y+baseline+5, x-5:x+text_w+5] = res
        
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def _draw_crosshair(self, img: np.ndarray, w: int, h: int):
        """Draws a crosshair in the center of the screen."""
        center_color = (0, 255, 255)  # Yellow
        cv2.circle(img, (w // 2, h // 2), 12, center_color, 2)
        cv2.line(img, (w // 2 - 8, h // 2), (w // 2 + 8, h // 2), center_color, 2)
        cv2.line(img, (w // 2, h // 2 - 8), (w // 2, h // 2 + 8), center_color, 2)

    def _draw_player_warnings(self, img: np.ndarray, player_targets: List[Dict[str, Any]]):
        """Draws bounding boxes and warnings for detected players."""
        if not player_targets: return
        for player in player_targets:
            x1, y1, x2, y2 = player['bbox']
            danger_color = (0, 0, 255)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), danger_color, 3)
            text = f"⚠ PLAYER {player['conf']:.2f}"
            self._draw_beautiful_text(img, text, (int(x1), int(y1) - 10), font_scale=0.6, color=danger_color)
            
            pulse = int(10 * (1 + 0.3 * np.sin(time.time() * 5)))
            cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), pulse, danger_color, 2)

    def _draw_current_target(self, img: np.ndarray, target: Optional[Dict[str, Any]], original_shape: Tuple[int, int, int]):
        """Draws an indicator for the bot's current target."""
        if not target: return
        h_orig, w_orig, _ = original_shape
        h_disp, w_disp, _ = img.shape

        tx = int(target['cx'] / w_orig * w_disp)
        ty = int(target['cy'] / h_orig * h_disp)
        target_color = (0, 255, 0)

        cv2.circle(img, (tx, ty), 15, target_color, 2)
        cv2.circle(img, (tx, ty), 8, target_color, -1)
        cv2.line(img, (w_disp // 2, h_disp // 2), (tx, ty), target_color, 2)
        
        dist = target.get('distance')
        conf = target.get('conf')
        dist_text = f"{dist:.0f}px" if isinstance(dist, (int, float)) else "N/A"
        conf_text = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
        info_text = f"TARGET: {dist_text} | {conf_text}"
        self._draw_beautiful_text(img, info_text, (tx + 20, ty - 10), font_scale=0.5, color=target_color)

    def _draw_hud(self, img: np.ndarray, bot_state: Any):
        """Draws the main Heads-Up Display with bot status."""
        y_offset, line_height = 30, 30

        # State
        state_color = (0, 255, 255) if bot_state.state == 'HUNTING' else (255, 255, 0)
        state_text = f"STATE: {bot_state.state}" + (f" ({bot_state.navigation_system.search_phase})" if bot_state.state == 'SEARCHING' else "")
        self._draw_beautiful_text(img, state_text, (10, y_offset), font_scale=0.7, color=state_color)
        
        # Info
        y_offset += line_height
        info_text = (
            f"FPS: {bot_state.frame_counter.fps:.1f} | Conf: {bot_state.conf_thres:.2f} | "
            f"Targets: {len(bot_state.last_known_candies[0]) if bot_state.last_known_candies else 0} | "
            f"Threats: {len(bot_state.confirmed_threats)}"
        )
        self._draw_beautiful_text(img, info_text, (10, y_offset), font_scale=0.6, color=(0, 255, 0))

        # Stats
        y_offset += line_height
        success_rate = (bot_state.stats['successful_collections'] / max(1, bot_state.stats['total_detections'])) * 100
        stats_text = f"Collections: {bot_state.stats['successful_collections']} | Success: {success_rate:.1f}%"
        self._draw_beautiful_text(img, stats_text, (10, y_offset), font_scale=0.6, color=(255, 255, 255))

        # Detection Pass
        y_offset += line_height
        if bot_state.last_detection_pass:
            pass_color = (255, 150, 0) if bot_state.last_detection_pass == 'heavy' else (150, 255, 150)
            pass_text = f"Detection: {bot_state.last_detection_pass.upper()}"
            self._draw_beautiful_text(img, pass_text, (10, y_offset), font_scale=0.6, color=pass_color)

    def _draw_fps_bar(self, img: np.ndarray, fps: float, w: int):
        """Draws a performance bar for FPS."""
        bar_x, bar_y, bar_width, bar_height = w - 210, 15, 200, 8
        
        fps_normalized = min(1.0, fps / 60.0)
        fps_color = (0, 255, 0) if fps_normalized > 0.5 else ((0, 255, 255) if fps_normalized > 0.3 else (0, 0, 255))
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * fps_normalized), bar_y + bar_height), fps_color, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

    def _draw_detected_targets(self, img: np.ndarray, targets: List[Dict[str, Any]], original_shape: Tuple[int, int, int]):
        """Draws indicators for all detected targets."""
        if not targets: return
        h_orig, w_orig, _ = original_shape
        h_disp, w_disp, _ = img.shape

        for i, target in enumerate(targets[:5]):
            tx = int(target['cx'] / w_orig * w_disp)
            ty = int(target['cy'] / h_orig * h_disp)
            priority = target.get('priority_score', 0)
            
            color = (0, 255, 0) if priority > 200 else ((0, 255, 255) if priority > 100 else (255, 255, 0))
            
            cv2.circle(img, (tx, ty), 10, color, 1)
            cv2.putText(img, f"#{i+1}", (tx + 12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def close(self):
        """Closes all OpenCV windows."""
        cv2.destroyAllWindows()
