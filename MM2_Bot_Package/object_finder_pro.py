"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  OBJECT FINDER PRO  —  Roblox Bot (обучение на роблоксе перед робототехникой)║
║  Работает без Shift Lock!                                                    ║
║  Поворот через win32api.mouse_event (относительное движение) —               ║
║  работает в оконном режиме, полноэкранном и с/без Shift Lock.                ║
╚══════════════════════════════════════════════════════════════════════════════╝

ЗАВИСИМОСТИ (установи через pip):
    pip install transformers torch torchvision supervision mss numpy opencv-python pywin32 keyboard colorama pyyaml scipy
    pip install dxcam  # опционально, для Win11/HDR-мониторов (+30% FPS)

ЗАПУСК:
    python coin_hunter_pro.py --weights PekingU/rtdetr_r50vd
    python coin_hunter_pro.py --weights PekingU/rtdetr_r50vd --show --fps-limit 60
    python coin_hunter_pro.py --weights PekingU/rtdetr_r50vd --config config.yaml

УПРАВЛЕНИЕ:
    Q или ESC  →  аварийная остановка
"""

# ─────────────────────────────────────────────────────────────── импорты ─────
import sys
import os
import time
import math
import random
import threading
import argparse
import queue
import json
import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import cv2
import keyboard                     # глобальное прослушивание клавиш

# Win32 API — управление мышью/клавишами на уровне ядра Windows
import win32api
import win32con

# AutoHotkey-обёртка для клавиш (та же, что использует рабочий Control)
try:
    from ahk import AHK
    ahk = AHK(executable_path='C:/Program Files/AutoHotkey/v2/AutoHotkey.exe')
    AHK_AVAILABLE = True
except Exception:
    AHK_AVAILABLE = False
    print("[WARN] AHK недоступен — будет использован win32api для клавиш")

# RT-DETR (Apache 2.0) — HuggingFace Transformers + Supervision ByteTrack
import torch
from PIL import Image as _PILImage
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import supervision as sv

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True, strip=False, convert=True)
except ImportError:
    class Fore:   GREEN = YELLOW = RED = CYAN = MAGENTA = BLUE = WHITE = RESET = ""
    class Style:  BRIGHT = RESET_ALL = ""

# scipy сплайны (опционально)
try:
    from scipy.interpolate import CubicSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# dxcam (опционально, Win11/HDR)
try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False

# mss — fallback захват
try:
    import mss as _mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# yaml для конфига
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ─────────────────────────────────────────────────── логирование ─────────────
_log_filename = f'bot_{datetime.now():%Y%m%d_%H%M%S}.log'
logging.basicConfig(
    filename=_log_filename,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger('CoinHunterPro')

# ─────────────────────────────────────────────────── глобальный флаг стопа ───
STOP = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
#  УТИЛИТА: Профилировщик
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def timer(name: str):
    """Контекстный менеджер для замера времени операций."""
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(f"{name}: {elapsed_ms:.2f}ms")


# ══════════════════════════════════════════════════════════════════════════════
#  СТАТИСТИКА СЕССИИ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionStats:
    coins_collected: int = 0
    session_start: float = field(default_factory=time.time)
    stuck_count: int = 0
    escapes: list = field(default_factory=list)

    def save(self):
        fname = f'session_{int(time.time())}.json'
        try:
            with open(fname, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            print(f"[STATS] Сохранено в {fname}")
        except Exception as e:
            log.error(f"SessionStats.save: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 1 — Движение мыши через mouse_event
# ══════════════════════════════════════════════════════════════════════════════

def mouse_move_relative(dx: int, dy: int):
    """
    Перемещает мышь на (dx, dy) пикселей ОТНОСИТЕЛЬНО текущей позиции
    через win32api.mouse_event с флагом MOUSEEVENTF_MOVE.

    dx > 0  →  поворот вправо
    dx < 0  →  поворот влево
    dy > 0  →  взгляд вниз
    dy < 0  →  взгляд вверх
    """
    if dx == 0 and dy == 0:
        return
    try:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
    except Exception:
        pass


def smooth_mouse_move(dx: int, dy: int, steps: int = 6, delay: float = 0.0008):
    """Плавно перемещает мышь на суммарно (dx, dy) пикселей за несколько шагов."""
    if dx == 0 and dy == 0:
        return
    step_x = dx / steps
    step_y = dy / steps
    accumulated_x = 0.0
    accumulated_y = 0.0
    try:
        for i in range(steps):
            accumulated_x += step_x
            accumulated_y += step_y
            send_x = int(accumulated_x)
            send_y = int(accumulated_y)
            if send_x != 0 or send_y != 0:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, send_x, send_y, 0, 0)
                accumulated_x -= send_x
                accumulated_y -= send_y
            if delay > 0:
                time.sleep(delay)
    except Exception:
        pass


def bezier_mouse_move(dx: int, dy: int, steps: int = 10, delay: float = 0.0008):
    """
    Движение мыши по кривой Безье — максимально человекоподобно.
    Используется для больших поворотов (escape-манёвры, random walk).
    """
    if dx == 0 and dy == 0:
        return
    cp_x = dx / 2 + random.randint(-abs(dx)//3 - 5, abs(dx)//3 + 5)
    cp_y = dy / 2 + random.randint(-15, 15)
    prev_x, prev_y = 0.0, 0.0
    try:
        for i in range(1, steps + 1):
            t = i / steps
            bx = 2*(1-t)*t * cp_x + t**2 * dx
            by = 2*(1-t)*t * cp_y + t**2 * dy
            send_x = int(bx - prev_x)
            send_y = int(by - prev_y)
            prev_x = bx - (send_x - (bx - prev_x - send_x))
            prev_y = by - (send_y - (by - prev_y - send_y))
            if send_x != 0 or send_y != 0:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, send_x, send_y, 0, 0)
            if delay > 0:
                time.sleep(delay)
    except Exception:
        pass


def spline_mouse_move(dx: int, dy: int, steps: int = 12, delay: float = 0.0007):
    """
    Движение мыши по кубическому сплайну (C2-непрерывность — плавнее Безье).
    Требует scipy. Если scipy недоступен — fallback на bezier_mouse_move.
    """
    if not SCIPY_AVAILABLE:
        bezier_mouse_move(dx, dy, steps, delay)
        return
    if dx == 0 and dy == 0:
        return
    try:
        t = np.array([0.0, 0.3, 0.7, 1.0])
        wx = np.array([0.0,
                       dx * 0.3 + random.randint(-20, 20),
                       dx * 0.7 + random.randint(-20, 20),
                       float(dx)])
        wy = np.array([0.0,
                       dy * 0.3 + random.randint(-10, 10),
                       dy * 0.7 + random.randint(-10, 10),
                       float(dy)])
        cs_x = CubicSpline(t, wx)
        cs_y = CubicSpline(t, wy)
        t_steps = np.linspace(0, 1, steps)
        xs = cs_x(t_steps)
        ys = cs_y(t_steps)
        prev_x, prev_y = 0.0, 0.0
        for x, y in zip(xs, ys):
            send_x = int(x - prev_x)
            send_y = int(y - prev_y)
            if send_x != 0 or send_y != 0:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, send_x, send_y, 0, 0)
            prev_x, prev_y = x, y
            if delay > 0:
                time.sleep(delay)
    except Exception as e:
        log.error(f"spline_mouse_move: {e}")
        bezier_mouse_move(dx, dy, steps, delay)


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 2 — PID-контроллер с плавным запуском интеграла
# ══════════════════════════════════════════════════════════════════════════════

class PIDController:
    """
    ПИД-регулятор с плавным сбросом интеграла при смене цели.

    После reset() первые `_reset_frames` кадров ki постепенно нарастает
    от 0 до ki — это предотвращает резкий рывок мыши при переключении цели.
    """
    def __init__(self, kp: float = 0.38, ki: float = 0.002, kd: float = 0.22,
                 integral_limit: float = 180.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self._integral      = 0.0
        self._prev_error    = 0.0
        self._prev_time     = time.time()
        self._reset_frames  = 5      # кадров плавного набора интеграла
        self._frame_counter = 0

    def compute(self, error: float) -> float:
        """
        Считает управляющий сигнал.
        error — отклонение цели от центра в пикселях.
        Возвращает: количество px для сдвига мыши.
        """
        now = time.time()
        dt  = max(0.001, now - self._prev_time)
        self._prev_time = now

        # Плавный запуск интеграла после reset()
        ki_eff = self.ki
        if self._frame_counter < self._reset_frames:
            ki_eff *= (self._frame_counter / max(1, self._reset_frames))
            self._frame_counter += 1

        # Интегральное накопление с ограничением (anti-windup)
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral + error * dt))

        # Производная — скорость изменения ошибки
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        return self.kp * error + ki_eff * self._integral + self.kd * derivative

    def reset(self):
        """Плавный сброс при смене цели (интеграл обнуляется, начинается warmup)."""
        self._integral      = 0.0
        self._prev_error    = 0.0
        self._prev_time     = time.time()
        self._frame_counter = 0


class AdaptivePID(PIDController):
    """
    PID с адаптивным kp: при большой ошибке — агрессивнее, при малой — плавнее.
    """
    def compute(self, error: float) -> float:
        scale = min(2.0, abs(error) / 100.0 + 0.5)
        saved_kp = self.kp
        self.kp = 0.38 * scale
        result = super().compute(error)
        self.kp = saved_kp
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 3 — Управление клавишами
# ══════════════════════════════════════════════════════════════════════════════

class KeyController:
    """
    Абстракция нажатий клавиш.
    Хранит состояние нажатых клавиш — не шлёт лишние key_down/key_up.
    """
    _VK = {
        'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44,
        'space': 0x20,
        'left': 0x25, 'right': 0x27, 'up': 0x26, 'down': 0x28,
    }
    _KEY_MAP = {
        'forward': 'w', 'back': 's', 'left': 'a', 'right': 'd',
        'jump': 'space',
    }

    def __init__(self):
        self._held: set = set()

    def _vk_down(self, k: str):
        vk = self._VK.get(k)
        if vk:
            win32api.keybd_event(vk, 0, 0, 0)

    def _vk_up(self, k: str):
        vk = self._VK.get(k)
        if vk:
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)

    def _send_down(self, k: str):
        if AHK_AVAILABLE:
            try:
                ahk.key_down(k, blocking=False)
                return
            except Exception:
                pass
        self._vk_down(k)

    def _send_up(self, k: str):
        if AHK_AVAILABLE:
            try:
                ahk.key_up(k, blocking=False)
                return
            except Exception:
                pass
        self._vk_up(k)

    def hold(self, action: str):
        """Удержать клавишу (только один key_down на нажатие)."""
        k = self._KEY_MAP.get(action, action)
        if k not in self._held:
            self._held.add(k)
            self._send_down(k)

    def release(self, action: str):
        """Отпустить клавишу."""
        k = self._KEY_MAP.get(action, action)
        if k in self._held:
            self._held.discard(k)
            self._send_up(k)

    def tap(self, action: str, duration: float = 0.08):
        """Кратковременное нажатие."""
        k = self._KEY_MAP.get(action, action)
        self._send_down(k)
        self._held.discard(k)
        time.sleep(duration)
        self._send_up(k)

    def release_all(self):
        """Отпустить все удерживаемые клавиши."""
        for k in list(self._held):
            self._send_up(k)
        self._held.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 4 — Захват экрана (dxcam / mss с поддержкой нескольких мониторов)
# ══════════════════════════════════════════════════════════════════════════════

class ScreenCapture:
    """
    Захват кадра через dxcam (предпочтительно — DirectX, +30% FPS на Win11/HDR)
    или mss (BitBlt Windows API) как fallback.

    Автоматически находит окно Roblox и определяет на каком мониторе оно.
    """
    def __init__(self, use_dxcam: bool = True, window_name: str = 'Roblox'):
        self._use_dxcam = use_dxcam and DXCAM_AVAILABLE
        self._camera = None
        self._window_name = window_name
        self.region  = self._find_window()

        if self._use_dxcam:
            try:
                self._camera = dxcam.create(output_color="BGR")
                x = self.region['left']
                y = self.region['top']
                w = self.region['width']
                h = self.region['height']
                self._camera.start(region=(x, y, x + w, y + h), target_fps=120)
                print("[CAPTURE] dxcam active (DirectX, HDR-compatible) [OK]")
                log.info("ScreenCapture: dxcam активен")
            except Exception as e:
                log.warning(f"dxcam init failed: {e}, fallback mss")
                self._use_dxcam = False
                self._camera = None

        if not self._use_dxcam:
            if not MSS_AVAILABLE:
                raise RuntimeError("Ни dxcam ни mss не доступны!")
            self.sct = _mss.mss()
            print("[CAPTURE] mss (BitBlt) active [OK]")
            log.info("ScreenCapture: mss активен")

    def _find_window(self) -> dict:
        try:
            import win32gui
            windows = []
            def cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    if self._window_name.lower() in win32gui.GetWindowText(hwnd).lower():
                        windows.append(hwnd)
            win32gui.EnumWindows(cb, None)
            if windows:
                hwnd = windows[0]
                x, y, x2, y2 = win32gui.GetWindowRect(hwnd)
                w, h = x2 - x, y2 - y
                monitor_idx = 1
                # Определяем на каком мониторе окно (для dxcam)
                if MSS_AVAILABLE:
                    with _mss.mss() as sct:
                        for i, mon in enumerate(sct.monitors[1:], 1):
                            if (mon['left'] <= x < mon['left'] + mon['width'] and
                                    mon['top'] <= y < mon['top'] + mon['height']):
                                monitor_idx = i
                                break
                region = {'top': y + 30, 'left': x, 'width': w, 'height': h - 30}
                print(f"[CAPTURE] {self._window_name}: {w}x{h} @ ({x},{y}), монитор #{monitor_idx}")
                log.info(f"Window {self._window_name}: {w}x{h} @ ({x},{y}), monitor #{monitor_idx}")
                return region
        except Exception as e:
            log.warning(f"_find_window: {e}")
        # Весь первый монитор как fallback
        if MSS_AVAILABLE:
            with _mss.mss() as sct:
                mon = sct.monitors[1]
                print(f"[CAPTURE] {self._window_name} не найден — весь экран {mon['width']}x{mon['height']}")
                return {'top': mon['top'], 'left': mon['left'],
                        'width': mon['width'], 'height': mon['height']}
        return {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

    def grab(self) -> np.ndarray:
        """Возвращает BGR numpy-массив текущего кадра."""
        if self._use_dxcam and self._camera is not None:
            frame = self._camera.get_latest_frame()
            if frame is not None:
                return frame
            return np.zeros((self.region['height'], self.region['width'], 3), np.uint8)
        # mss fallback
        raw = self.sct.grab(self.region)
        return np.array(raw, dtype=np.uint8)[:, :, :3]

    def __del__(self):
        if self._use_dxcam and self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.region['height'], self.region['width'])


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 5 — Тепловая карта монеток
# ══════════════════════════════════════════════════════════════════════════════

class CoinHeatmap:
    """
    Запоминает где чаще всего появляются монетки.
    Используется в SEARCH-режиме для выбора направления поиска.
    """
    def __init__(self, grid: int = 20, decay: float = 0.999):
        self.grid  = grid
        self.decay = decay
        self._map  = np.zeros((grid, grid), dtype=np.float32)

    def record(self, cx: float, cy: float, fw: int, fh: int):
        gx = int(cx / fw * self.grid)
        gy = int(cy / fh * self.grid)
        gx = max(0, min(self.grid - 1, gx))
        gy = max(0, min(self.grid - 1, gy))
        self._map[gy, gx] += 1.0
        self._map *= self.decay

    def best_direction(self) -> Tuple[int, int]:
        """Возвращает (dx, dy) мыши в сторону самой горячей зоны."""
        y, x = np.unravel_index(self._map.argmax(), self._map.shape)
        cx = self.grid // 2
        return (x - cx) * 15, 0


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 6 — Детектор RT-DETR + ByteTrack + Ghost tracking  (Apache 2.0)
# ══════════════════════════════════════════════════════════════════════════════

class CoinDetector:
    """
    RT-DETR (Apache 2.0) с ByteTrack (supervision, Apache 2.0) и Ghost Tracking.

    Ghost Tracking: если монетка исчезла на <0.5с (окклюзия) — сохраняем
    последние координаты как «призрак» и продолжаем наводиться на него.
    Это предотвращает смену ID и дёрганье бота при кратковременных потерях.

    Адаптивный порог conf: если RT-DETR видит >8 объектов — вероятно ложные
    срабатывания, повышаем порог. Если <2 — снижаем.
    """
    MIN_BOX_PX     = 6
    MAX_BOX_PX     = 400  # Increased from 160 to detect larger coin groups
    MIN_ASPECT     = 0.35
    MAX_ASPECT     = 2.8
    MAX_Y_FRAC     = 0.90
    MIN_Y_FRAC     = 0.20
    CONFIRM_FRAMES = 2
    GHOST_TTL      = 1.5    # секунд — максимум хранить призрак
    GHOST_USE_FOR  = 0.5    # секунд — использовать призрак как цель

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 320,
                 coin_class_id: int = 1):
        print(f"[RT-DETR] Загрузка модели: {weights}")
        log.info(f"CoinDetector init: {weights}, conf={conf}, imgsz={imgsz}")
        # Normalize local path - remove './' prefix to avoid HuggingFace treating it as repo ID
        if weights.startswith('./'):
            weights = weights[2:]
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor      = RTDetrImageProcessor.from_pretrained(weights)
        self.model          = RTDetrForObjectDetection.from_pretrained(weights).to(self.device)
        self.model.eval()
        self.conf_min       = conf
        self.conf_adaptive  = conf
        self.imgsz          = imgsz
        self.coin_class_id  = coin_class_id
        # Буфер подтверждения
        self._confirm: dict = {}
        # Ghost tracking
        self._ghosts: dict  = {}   # {id: {'cx','cy','last_seen'}}
        self._false_positive_count = 0
        # Тепловая карта
        self.heatmap = CoinHeatmap()
        # ByteTrack через supervision (Apache 2.0)
        self._tracker = sv.ByteTrack()
        self._next_id = 0
        # JIT-прогрев
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self._run_inference(dummy)
        print("[RT-DETR] Model ready [OK]")
        log.info("RT-DETR model ready")

    def _run_inference(self, frame: np.ndarray):
        """Запускает RT-DETR инференс, возвращает supervision Detections."""
        h, w = frame.shape[:2]
        pil_img = _PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # post_process возвращает список [{scores, labels, boxes}]
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.conf_adaptive, target_sizes=target_sizes
        )[0]
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes  = results["boxes"].cpu().numpy()   # xyxy

        # DEBUG: Show all detections before filtering
        if len(boxes) > 0:
            print(f"[DEBUG] Raw detections: {len(boxes)} objects, classes={np.unique(labels)}")
            for i in range(min(3, len(boxes))):
                print(f"  - class={labels[i]}, conf={scores[i]:.2f}, box={boxes[i]}")

        # Фильтруем по классу
        mask = labels == self.coin_class_id
        scores = scores[mask]
        boxes  = boxes[mask]

        if len(boxes) == 0:
            return sv.Detections.empty()

        sv_dets = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=np.full(len(boxes), self.coin_class_id, dtype=int),
        )
        return sv_dets

    def _is_valid_coin(self, cx: float, cy: float, w: float, h: float,
                       fw: int, fh: int) -> bool:
        if w < self.MIN_BOX_PX or h < self.MIN_BOX_PX:
            return False
        if w > self.MAX_BOX_PX or h > self.MAX_BOX_PX:
            return False
        aspect = w / max(h, 1)
        if not (self.MIN_ASPECT <= aspect <= self.MAX_ASPECT):
            return False
        y_frac = cy / max(fh, 1)
        if not (self.MIN_Y_FRAC <= y_frac <= self.MAX_Y_FRAC):
            return False
        return True

    def detect(self, frame: np.ndarray) -> list:
        """
        Запускает детекцию+трекинг RT-DETR с ghost tracking и адаптивным conf.
        Возвращает список словарей: id, cx, cy, w, h, conf, dist, ghost
        """
        try:
            fh, fw = frame.shape[:2]
            cx_sc, cy_sc = fw / 2, fh / 2

            sv_dets = self._run_inference(frame)
            # ByteTrack через supervision
            tracked = self._tracker.update_with_detections(sv_dets)

            seen_ids: set = set()
            coins: list   = []
            now = time.time()

            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                bw   = x2 - x1
                bh   = y2 - y1
                bcx  = (x1 + x2) / 2
                bcy  = (y1 + y2) / 2
                tid  = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                conf_val = float(tracked.confidence[i]) if tracked.confidence is not None else self.conf_adaptive

                if not self._is_valid_coin(bcx, bcy, bw, bh, fw, fh):
                    self._confirm.pop(tid, None)
                    continue

                seen_ids.add(tid)
                count = self._confirm.get(tid, 0) + 1
                self._confirm[tid] = count
                if count < self.CONFIRM_FRAMES:
                    continue

                coins.append({
                    'id':    tid,
                    'cx':    bcx,
                    'cy':    bcy,
                    'w':     bw,
                    'h':     bh,
                    'conf':  conf_val,
                    'dist':  math.hypot(bcx - cx_sc, bcy - cy_sc),
                    'frames': count,
                    'ghost': False,
                })
                # Обновляем тепловую карту
                self.heatmap.record(bcx, bcy, fw, fh)

            # ── Адаптивный confidence ──────────────────────────────────────
            if len(coins) > 8:
                self.conf_adaptive = min(0.50, self.conf_adaptive + 0.02)
                self._false_positive_count += 1
                if self._false_positive_count > 10:
                    log.warning(f"Много ложных целей, conf={self.conf_adaptive:.2f}")
            elif len(coins) < 2:
                self.conf_adaptive = max(self.conf_min, self.conf_adaptive - 0.01)

            # ── Обновляем призраков ────────────────────────────────────────
            # Удаляем устаревших
            self._ghosts = {k: v for k, v in self._ghosts.items()
                            if now - v['last_seen'] < self.GHOST_TTL}

            # Сохраняем текущие монетки как призраков
            for c in coins:
                self._ghosts[c['id']] = {
                    'cx': c['cx'], 'cy': c['cy'],
                    'last_seen': now
                }

            # Добавляем свежих призраков (пропавшие <0.5c)
            for cid, ghost in self._ghosts.items():
                age = now - ghost['last_seen']
                if cid not in seen_ids and age < self.GHOST_USE_FOR:
                    coins.append({
                        'id':    cid,
                        'cx':    ghost['cx'],
                        'cy':    ghost['cy'],
                        'w':     20,
                        'h':     20,
                        'conf':  0.3,
                        'dist':  math.hypot(ghost['cx'] - cx_sc, ghost['cy'] - cy_sc),
                        'frames': 1,
                        'ghost': True,
                    })

            # Удаляем из буфера подтверждения устаревшие ID
            stale = [k for k in self._confirm if k not in seen_ids]
            for k in stale:
                del self._confirm[k]

            return coins

        except Exception as e:
            log.error(f"CoinDetector.detect: {e}", exc_info=True)
            return []

    def pick_nearest(self, coins: list, fw: int = 640) -> Optional[dict]:
        """
        Выбирает лучшую цель с учётом:
        - расстояния
        - вертикального положения (ближе к игроку = ниже)
        - бокового отклонения (монетка прямо перед игроком важнее)
        - числа кадров подтверждения
        - уверенности RT-DETR
        - призраки имеют штраф
        """
        if not coins:
            return None
        def score(c):
            y_bonus       = max(0.0, (c['cy'] - 0.5)) * 0.35
            conf_bonus    = min(0.15, (c['conf'] - 0.25) * 0.3)
            frames_bonus  = min(0.15, (c.get('frames', 1) - 1) * 0.03)
            # Штраф за боковое отклонение
            lateral_pen   = abs(c['cx'] - fw / 2) / fw * 0.5
            ghost_penalty = 0.3 if c.get('ghost') else 0.0
            return c['dist'] * (1.0 + lateral_pen + ghost_penalty
                                - y_bonus - conf_bonus - frames_bonus)
        return min(coins, key=score)


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 7 — Random Walk с памятью поворотов
# ══════════════════════════════════════════════════════════════════════════════

class RandomWalker:
    """
    Случайное блуждание в поиске монеток.

    Улучшение: _turn_history запоминает последние 3 поворота.
    Если бот повернул влево 2+ раза подряд — принудительно поворачивает вправо,
    предотвращая зацикливание в углу карты.
    """
    def __init__(self, keys: KeyController):
        self.keys          = keys
        self._phase        = 'WALK'
        self._phase_end    = 0.0
        self._strafe_end   = 0.0
        self._strafe_key   = ''
        self._jump_timer   = 0.0
        self._turn_history = deque(maxlen=3)  # +1=вправо, -1=влево

    def update(self):
        """Вызывается каждый кадр в режиме SEARCH."""
        now = time.time()

        if now > self._jump_timer:
            if random.random() < 0.015:
                self.keys.tap('jump', 0.05)
            self._jump_timer = now + random.uniform(1.5, 4.0)

        if now >= self._phase_end:
            self._next_phase(now)

        if self._phase == 'WALK':
            self.keys.hold('forward')
            if now < self._strafe_end and self._strafe_key:
                self.keys.hold(self._strafe_key)
            else:
                self.keys.release('left')
                self.keys.release('right')
                if random.random() < 0.25 and now >= self._strafe_end:
                    self._strafe_key = random.choice(['left', 'right'])
                    self._strafe_end = now + random.uniform(0.15, 0.45)
        elif self._phase == 'TURN':
            self.keys.release('forward')
            self.keys.release('left')
            self.keys.release('right')

    def _next_phase(self, now: float):
        if self._phase == 'WALK':
            self._phase     = 'TURN'
            self._phase_end = now + random.uniform(0.3, 0.7)

            # Баланс последних поворотов (-1=лево, +1=право)
            balance = sum(self._turn_history)
            if balance < -1.5:
                # Слишком много левых поворотов — тянем вправо
                angle_px = random.randint(60, 140)
            elif balance > 1.5:
                # Слишком много правых — тянем влево
                angle_px = -random.randint(60, 140)
            else:
                angle_px = random.randint(60, 200) * random.choice([-1, 1])

            self._turn_history.append(1 if angle_px > 0 else -1)
            bezier_mouse_move(angle_px, random.randint(-10, 10),
                              steps=12, delay=0.003)
        else:
            self._phase     = 'WALK'
            self._phase_end = now + random.uniform(1.2, 3.5)

    def stop(self):
        self.keys.release('forward')
        self.keys.release('left')
        self.keys.release('right')


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 8 — Навигация (AdaptivePID + оптический поток + улучшенный escape)
# ══════════════════════════════════════════════════════════════════════════════

class NavigationPro:
    """
    Главный модуль движения.

    Изменения v2:
    - mouse_dy инвертирован: error_y * -0.12 (монеты на полу → смотрим вниз правильно)
    - AdaptivePID (масштабируемый kp)
    - Escape: cooldown 1.5с, случайная стадия, агрессивный разворот (stage 3)
    - Optical flow для детекции застревания
    - Использует CoinHeatmap в SEARCH-режиме
    """
    AIM_DEADZONE_PX   = 20
    COLLECT_DIST_PX   = 55
    MOUSE_SENSITIVITY = 0.40
    MOUSE_MAX_STEP    = 55
    STRAFE_THRESHOLD  = 0.22
    STRAFE_CHANCE     = 0.06
    JUMP_CHANCE       = 0.012
    STUCK_SECONDS     = 5.0
    DEBOUNCE_SEC      = 3.0
    ESCAPE_COOLDOWN   = 1.5   # было 3.5 — теперь быстрее реагируем

    def __init__(self, keys: KeyController):
        self.keys   = keys
        self.pid    = AdaptivePID(kp=0.38, ki=0.002, kd=0.22)
        self.walker = RandomWalker(keys)
        self.stats  = SessionStats()

        self._smooth_cx: Optional[float] = None
        self._smooth_cy: Optional[float] = None
        self._smooth_alpha = 0.38

        self._dist_history:       deque = deque(maxlen=30)
        self._last_progress_time: float = time.time()
        self._last_escape_time:   float = 0.0

        self._collected_ids: dict = {}
        self._target_id:     int  = -1

        # Optical flow для детекции застревания
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._low_flow_streak: int = 0

        self.coins_collected: int = 0

    # ─── Публичный API ────────────────────────────────────────────────────────

    def update(self, coin: Optional[dict], frame_shape: Tuple[int, int],
               frame: Optional[np.ndarray] = None,
               heatmap: Optional[CoinHeatmap] = None) -> str:
        """
        Вызывается каждый кадр.
        frame — текущий кадр (для optical flow, опционально).
        Возвращает строку статуса для HUD.
        """
        try:
            # Optical flow stuck check
            if frame is not None:
                self._update_optical_flow(frame)

            if coin is None:
                self._smooth_cx = None
                self._smooth_cy = None
                self.pid.reset()
                # Направление поиска по тепловой карте
                if heatmap is not None:
                    dx, dy = heatmap.best_direction()
                    if abs(dx) > 5:
                        smooth_mouse_move(dx, 0, steps=3, delay=0.001)
                self.walker.update()
                return "SEARCH"

            self.walker.stop()

            fh, fw    = frame_shape
            cx_screen = fw / 2

            cid = coin['id']
            if cid in self._collected_ids:
                if time.time() - self._collected_ids[cid] < self.DEBOUNCE_SEC:
                    return "DEBOUNCE"
                del self._collected_ids[cid]

            if cid != self._target_id:
                self.pid.reset()
                self._smooth_cx = None
                self._smooth_cy = None
                self._target_id = cid

            if self._smooth_cx is None:
                self._smooth_cx = coin['cx']
                self._smooth_cy = coin['cy']
            else:
                a = self._smooth_alpha
                self._smooth_cx = a * coin['cx'] + (1 - a) * self._smooth_cx
                self._smooth_cy = a * coin['cy'] + (1 - a) * self._smooth_cy

            error_x = self._smooth_cx - cx_screen
            error_y = self._smooth_cy - (fh / 2)
            dist    = coin['dist']

            if dist < self.COLLECT_DIST_PX:
                self.coins_collected += 1
                self.stats.coins_collected += 1
                self._collected_ids[cid] = time.time()
                self.keys.release_all()
                log.info(f"Монетка #{self.coins_collected} собрана (ID={cid})")
                print(f"\n[$] Monetka #{self.coins_collected} collected! (ID={cid})")
                return "COLLECTED"

            self._dist_history.append(dist)
            if self._is_stuck():
                return self._escape()

            if abs(error_x) > self.AIM_DEADZONE_PX:
                pid_out  = self.pid.compute(error_x)
                mouse_dx = int(max(-self.MOUSE_MAX_STEP,
                                   min(self.MOUSE_MAX_STEP, pid_out * self.MOUSE_SENSITIVITY)))
                # ✅ ИСПРАВЛЕНИЕ: инверсия dy — монетки на полу (error_y > 0)
                # требуют взгляда ВНИЗ (dy > 0), множитель отрицательный ошибочен.
                # Правильная формула: чем монетка ниже центра, тем сильнее опускаем взгляд.
                # Было: error_y * 0.10 (опускало, но по неправильной логике)
                # Стало: error_y * 0.12 (чуть агрессивнее, правильное направление)
                mouse_dy = int(max(-20, min(20, error_y * 0.12)))
                smooth_mouse_move(mouse_dx, mouse_dy, steps=4, delay=0.0007)
            else:
                self.pid.reset()

            self.keys.hold('forward')

            if abs(error_x) > fw * self.STRAFE_THRESHOLD:
                if error_x > 0:
                    self.keys.hold('right'); self.keys.release('left')
                else:
                    self.keys.hold('left');  self.keys.release('right')
            else:
                if random.random() < self.STRAFE_CHANCE:
                    sk = 'left' if random.random() < 0.5 else 'right'
                    self.keys.hold(sk)
                    threading.Timer(0.07, lambda s=sk: self.keys.release(s)).start()
                else:
                    self.keys.release('left')
                    self.keys.release('right')

            if error_y < -55 and dist < 180:
                self.keys.tap('jump', 0.05)
            elif random.random() < self.JUMP_CHANCE:
                self.keys.tap('jump', 0.04)

            ghost_tag = " [GHOST]" if coin.get('ghost') else ""
            return f"> COIN#{cid}  dx={error_x:+.0f}px  dist={dist:.0f}px{ghost_tag}"

        except Exception as e:
            log.error(f"NavigationPro.update crashed: {e}", exc_info=True)
            return "ERROR"

    # ─── Оптический поток ─────────────────────────────────────────────────────

    def _update_optical_flow(self, frame: np.ndarray):
        """Проверяет, движется ли игрок (для дополнительной детекции застревания)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._prev_frame_gray is not None and self._target_id != -1:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_frame_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0)
                mean_flow = float(np.mean(np.abs(flow)))
                if mean_flow < 0.5:
                    self._low_flow_streak += 1
                else:
                    self._low_flow_streak = 0
            self._prev_frame_gray = gray
        except Exception:
            pass

    # ─── Внутренние методы ────────────────────────────────────────────────────

    def _is_stuck(self) -> bool:
        """Расстояние не уменьшается ИЛИ оптический поток слишком мал."""
        if len(self._dist_history) < 20:
            return False
        progress = self._dist_history[0] - self._dist_history[-1]
        if progress > 5.0:
            self._last_progress_time = time.time()
            self._low_flow_streak = 0
            return False
        # Optical flow: нет движения >60 кадров при нажатом W
        if self._low_flow_streak > 60:
            log.warning("Застрял по optical flow!")
            return True
        return (time.time() - self._last_progress_time) > self.STUCK_SECONDS

    def _escape(self) -> str:
        """
        Escape-манёвр. Cooldown уменьшен до 1.5c.
        Стадия выбирается случайно (не последовательно) — лучше выбраться.
        """
        now = time.time()
        if now - self._last_escape_time < self.ESCAPE_COOLDOWN:
            return "STUCK_WAIT"
        self._last_escape_time = now
        self.keys.release_all()

        # Случайный манёвр (был последовательный — теперь случайный)
        stage = random.choice([0, 1, 2, 3])
        self.stats.stuck_count += 1
        self.stats.escapes.append({'time': now, 'stage': stage})
        log.info(f"Escape стадия {stage}")
        print(f"\n[STUCK] Escape стадия {stage + 1}...")

        try:
            if stage == 0:
                # Назад + страф
                self.keys.tap('back', 0.4)
                self.keys.tap(random.choice(['left', 'right']), 0.5)

            elif stage == 1:
                # Страф + прыжок
                k = random.choice(['left', 'right'])
                self.keys.hold('forward')
                self.keys.hold(k)
                self.keys.tap('jump', 0.08)
                time.sleep(0.3)
                self.keys.release_all()

            elif stage == 2:
                # Назад + поворот мыши
                self.keys.tap('back', 0.5)
                dx = random.choice([-1, 1]) * random.randint(80, 140)
                bezier_mouse_move(dx, 0, steps=10, delay=0.003)

            else:
                # ✅ Разворот на ~180° — гарантированный выход
                self.keys.tap('back', 0.45)
                self.keys.tap('jump', 0.05)
                dx = random.choice([-1, 1]) * random.randint(280, 320)
                spline_mouse_move(dx, 0, steps=8)
                self.keys.tap('forward', 0.6)

        except Exception as e:
            log.error(f"_escape: {e}")

        self._dist_history.clear()
        self._last_progress_time = time.time()
        self._low_flow_streak = 0
        self.pid.reset()
        return "ESCAPE"


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 9 — Консольный HUD
# ══════════════════════════════════════════════════════════════════════════════

class ConsoleHUD:
    """Лёгкий in-place HUD в терминале."""
    def __init__(self, update_every: int = 12):
        self._n        = update_every
        self._frame    = 0
        self._fps_hist = deque(maxlen=30)
        self._last_t   = time.time()

    def tick(self, status: str, coins: int, target: Optional[dict]):
        self._frame += 1
        now = time.time()
        self._fps_hist.append(1.0 / max(1e-6, now - self._last_t))
        self._last_t = now
        if self._frame % self._n != 0:
            return
        fps = sum(self._fps_hist) / len(self._fps_hist)
        ghost_tag = " 👻" if (target and target.get('ghost')) else ""
        tgt = (f"ID={target['id']}  dist={target['dist']:.0f}px{ghost_tag}"
               if target else "нет монетки       ")
        print(
            f"\r"
            f"{Fore.CYAN}FPS:{fps:5.1f}{Style.RESET_ALL}  "
            f"{Fore.YELLOW}COINS {coins:4d}{Style.RESET_ALL}  "
            f"{Fore.GREEN}Цель: {tgt:<34}{Style.RESET_ALL}  "
            f"[{status:<30}]",
            end='', flush=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 10 — Авто-рестарт (watchdog)
# ══════════════════════════════════════════════════════════════════════════════

def is_roblox_running() -> bool:
    try:
        import subprocess
        out = subprocess.check_output('tasklist /FI "IMAGENAME eq RobloxPlayerBeta.exe"',
                                      shell=True).decode('utf-8', errors='ignore')
        return 'RobloxPlayerBeta.exe' in out
    except Exception:
        return True

def auto_restart_watcher(interval: float = 10.0):
    while not STOP.is_set():
        time.sleep(interval)
        if not is_roblox_running():
            print("\n[WATCHDOG] Roblox закрылся — останавливаем бота")
            log.info("Roblox process not found — stopping bot")
            STOP.set()


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 11 — Параллельный pipeline захват→инференс
# ══════════════════════════════════════════════════════════════════════════════

class ParallelPipeline:
    """
    Разделяет захват кадра и инференс RT-DETR на два потока.
    GPU (инференс) и CPU (захват+навигация) работают параллельно → +20–40% FPS.
    """
    def __init__(self, screen: ScreenCapture, detector: CoinDetector):
        self._screen   = screen
        self._detector = detector
        self._frame_q  = queue.Queue(maxsize=2)
        self._result_q = queue.Queue(maxsize=2)
        self._t_capture   = threading.Thread(target=self._capture_loop, daemon=True)
        self._t_inference = threading.Thread(target=self._inference_loop, daemon=True)

    def start(self):
        self._t_capture.start()
        self._t_inference.start()
        log.info("ParallelPipeline started")

    def _capture_loop(self):
        while not STOP.is_set():
            frame = self._screen.grab()
            if not self._frame_q.full():
                self._frame_q.put(frame)

    def _inference_loop(self):
        while not STOP.is_set():
            try:
                frame = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            coins = self._detector.detect(frame)
            if not self._result_q.full():
                self._result_q.put((frame, coins))

    def get_latest(self) -> Tuple[Optional[np.ndarray], list]:
        """Возвращает последний(frame, coins) или (None, []) если нет данных."""
        result = None
        while not self._result_q.empty():
            result = self._result_q.get_nowait()
        return result if result else (None, [])


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 12 — Загрузка конфига
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    """Загружает YAML-конфиг и возвращает словарь параметров."""
    if not YAML_AVAILABLE:
        log.warning("PyYAML не установлен — конфиг проигнорирован")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        log.info(f"Конфиг загружен из {path}")
        return cfg
    except Exception as e:
        log.error(f"load_config({path}): {e}")
        return {}

def apply_config(cfg: dict, nav: NavigationPro, detector: CoinDetector):
    """Применяет параметры из конфига к объектам бота."""
    nav_cfg = cfg.get('navigation', {})
    det_cfg = cfg.get('detector', {})

    if 'aim_deadzone_px'   in nav_cfg: nav.AIM_DEADZONE_PX   = nav_cfg['aim_deadzone_px']
    if 'collect_dist_px'   in nav_cfg: nav.COLLECT_DIST_PX   = nav_cfg['collect_dist_px']
    if 'mouse_sensitivity' in nav_cfg: nav.MOUSE_SENSITIVITY  = nav_cfg['mouse_sensitivity']
    if 'stuck_seconds'     in nav_cfg: nav.STUCK_SECONDS      = nav_cfg['stuck_seconds']
    if 'min_box_px'        in det_cfg: detector.MIN_BOX_PX    = det_cfg['min_box_px']
    if 'max_box_px'        in det_cfg: detector.MAX_BOX_PX    = det_cfg['max_box_px']
    if 'confirm_frames'    in det_cfg: detector.CONFIRM_FRAMES = det_cfg['confirm_frames']
    log.info(f"Конфиг применён: nav={nav_cfg}, det={det_cfg}")


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 13 — Главный цикл
# ══════════════════════════════════════════════════════════════════════════════

def run_bot(weights: str, conf: float, imgsz: int, coin_class: int,
            show_cv: bool, fps_limit: int, config_path: str,
            use_dxcam: bool, parallel: bool, window_name: str = 'Roblox'):
    """Точка входа — инициализация и главный loop."""

    print(f"\n{Fore.CYAN}{'='*62}")
    print("  COIN HUNTER PRO v2 — старт")
    print(f"  Модель:  {weights}")
    print(f"  Порог:   {conf}  |  imgsz: {imgsz}")
    print(f"  Окно:    {window_name}")
    print(f"  Захват:  {'dxcam' if (use_dxcam and DXCAM_AVAILABLE) else 'mss'}")
    print(f"  Pipeline: {'параллельный' if parallel else 'последовательный'}")
    print(f"  Лог:     {_log_filename}")
    print(f"  Стоп:    Q или ESC")
    print(f"{'='*62}{Style.RESET_ALL}\n")

    log.info(f"Bot started: weights={weights}, conf={conf}, imgsz={imgsz}, window={window_name}")

    keyboard.add_hotkey('q',   lambda: STOP.set())
    keyboard.add_hotkey('esc', lambda: STOP.set())

    watcher = threading.Thread(target=auto_restart_watcher, daemon=True)
    watcher.start()

    screen   = ScreenCapture(use_dxcam=use_dxcam, window_name=window_name)
    detector = CoinDetector(weights, conf=conf, imgsz=imgsz, coin_class_id=coin_class)
    keys     = KeyController()
    nav      = NavigationPro(keys)
    hud      = ConsoleHUD(update_every=12)

    # Конфиг
    if config_path:
        cfg = load_config(config_path)
        if cfg:
            apply_config(cfg, nav, detector)

    min_frame_dt = 1.0 / fps_limit if fps_limit > 0 else 0.0
    last_t       = time.time()

    print("[BOT] Переключитесь на окно Roblox. Старт через 2 секунды...")
    time.sleep(2.0)
    print("[BOT] Пошёл!\n")

    # Параллельный pipeline
    pipeline: Optional[ParallelPipeline] = None
    if parallel:
        pipeline = ParallelPipeline(screen, detector)
        pipeline.start()

    try:
        while not STOP.is_set():
            t0 = time.time()
            elapsed = t0 - last_t
            if elapsed < min_frame_dt:
                time.sleep(min_frame_dt - elapsed)
            last_t = time.time()

            if pipeline is not None:
                # Параллельный режим: берём готовый результат
                frame, coins = pipeline.get_latest()
                if frame is None:
                    time.sleep(0.005)
                    continue
                fh, fw = frame.shape[:2]
            else:
                # Последовательный режим
                with timer("Capture"):
                    frame = screen.grab()
                fh, fw = frame.shape[:2]
                with timer("RT-DETR"):
                    coins = detector.detect(frame)

            target = detector.pick_nearest(coins, fw=fw)

            with timer("Navigation"):
                status = nav.update(target, (fh, fw),
                                    frame=frame,
                                    heatmap=detector.heatmap)

            hud.tick(status, nav.coins_collected, target)

            if show_cv:
                _draw_debug(frame, coins, target, fh, fw)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[BOT] Стоп. Собрано монет: {nav.coins_collected}")
        log.info(f"Bot stopped. Coins collected: {nav.coins_collected}")
        keys.release_all()
        nav.stats.save()
        if show_cv:
            cv2.destroyAllWindows()


def _draw_debug(frame: np.ndarray, coins: list, target: Optional[dict],
                fh: int, fw: int):
    """OpenCV-окно с боксами (только при --show)."""
    dbg = frame.copy()
    for c in coins:
        x1 = int(c['cx'] - c['w'] / 2)
        y1 = int(c['cy'] - c['h'] / 2)
        x2 = int(c['cx'] + c['w'] / 2)
        y2 = int(c['cy'] + c['h'] / 2)
        is_target = target and c['id'] == target['id']
        is_ghost  = c.get('ghost', False)
        if is_target:
            color = (0, 255, 255)
        elif is_ghost:
            color = (128, 0, 255)   # фиолетовый для призраков
        else:
            color = (100, 100, 100)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
        label = f"#{c['id']} {c['conf']:.2f}" + (" G" if is_ghost else "")
        cv2.putText(dbg, label, (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.line(dbg, (fw//2-18, fh//2), (fw//2+18, fh//2), (0, 255, 0), 1)
    cv2.line(dbg, (fw//2, fh//2-18), (fw//2, fh//2+18), (0, 255, 0), 1)
    cv2.circle(dbg, (fw//2, fh//2), 5, (0, 255, 0), 1)
    cv2.imshow("CoinHunterPro v2 — Debug [Q=выход]", dbg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        STOP.set()


# ══════════════════════════════════════════════════════════════════════════════
#  СЕКЦИЯ 14 — Точка входа (argparse)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Coin Hunter Pro v2 — Roblox MM2 Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python coin_hunter_pro.py --weights weights/balls+coins.pt
  python coin_hunter_pro.py --weights weights/balls+coins.pt --show --conf 0.3
  python coin_hunter_pro.py --weights weights/balls+coins.pt --dxcam --parallel
  python coin_hunter_pro.py --weights weights/balls+coins.pt --config config.yaml
        """
    )
    parser.add_argument('--weights',    default='PekingU/rtdetr_r50vd',
                        help='HuggingFace model id или локальный путь (RT-DETR, Apache 2.0)')
    parser.add_argument('--conf',       type=float, default=0.25,
                        help='Порог уверенности RT-DETR')
    parser.add_argument('--imgsz',      type=int,   default=640,
                        help='Размер входа RT-DETR (640 стандарт)')
    parser.add_argument('--coin-class', type=int,   default=1,
                        help='ID класса монетки в модели')
    parser.add_argument('--show',       action='store_true',
                        help='Показывать OpenCV-окно с детекцией')
    parser.add_argument('--fps-limit',  type=int,   default=60,
                        help='Максимальный FPS бота (0 = без ограничений)')
    parser.add_argument('--config',     default='',
                        help='Путь к YAML-конфигу (опционально)')
    parser.add_argument('--window',     default='Roblox',
                        help='Название окна для захвата (по умолчанию: Roblox)')
    parser.add_argument('--dxcam',      action='store_true',
                        help='Использовать dxcam вместо mss (Win11/HDR, +30%% FPS)')
    parser.add_argument('--parallel',   action='store_true',
                        help='Параллельный захват+инференс (GPU+CPU одновременно)')
    args = parser.parse_args()

    run_bot(
        weights     = args.weights,
        conf        = args.conf,
        imgsz       = args.imgsz,
        coin_class  = args.coin_class,
        show_cv     = args.show,
        fps_limit   = args.fps_limit,
        config_path = args.config,
        use_dxcam   = args.dxcam,
        parallel    = args.parallel,
        window_name = args.window,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ: КАК ЭТО ПЕРЕНЕСТИ В РЕАЛЬНУЮ РОБОТОТЕХНИКУ
#  (Raspberry Pi 5 / Arduino + камера + RT-DETR + PID для моторов)
# ══════════════════════════════════════════════════════════════════════════════
"""
═══════════════════════════════════════════════════════════════════════════════
  МОСТ: ОТ ROBLOX-БОТА К РЕАЛЬНОМУ РОБОТУ
═══════════════════════════════════════════════════════════════════════════════

1. ЗАХВАТ КАМЕРЫ  (аналог mss)
───────────────────────────────
Roblox:   frame = screen.grab()              # mss/dxcam с экрана
Робот:    cap = cv2.VideoCapture(0)          # USB/CSI камера
          ret, frame = cap.read()             # тот же numpy BGR-массив!

  Raspberry Pi + picamera2:
    from picamera2 import Picamera2
    cam = Picamera2(); cam.start()
    frame = cam.capture_array()              # RGBA → конвертируй в BGR


2. ДЕТЕКЦИЯ  (RT-DETR Apache 2.0 — идентичен на ПК и Pi)
──────────────────────────────────────────────────────────
  from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
  processor = RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_r18vd')  # лёгкая для Pi
  model = RTDetrForObjectDetection.from_pretrained('PekingU/rtdetr_r18vd')
  # Для Pi 5: экспортируй в ONNX (~2× быстрее)
  # model.save_pretrained('rtdetr_r18vd_local')


3. PID ДЛЯ МОТОРОВ  (аналог smooth_mouse_move)
───────────────────────────────────────────────
Roblox:   smooth_mouse_move(pid_out)         # поворот камеры
Робот:    pid_out → скорость поворота мотора

  Arduino (pyserial):
    import serial
    arduino = serial.Serial('/dev/ttyUSB0', 9600)
    error_x  = coin_cx - frame_w / 2        # та же формула!
    pid_out  = pid.compute(error_x)          # тот же PID-класс!
    left_pwm  = base_speed - pid_out
    right_pwm = base_speed + pid_out
    arduino.write(f"M{int(left_pwm)},{int(right_pwm)}\n".encode())

  Raspberry Pi + RPi.GPIO:
    left_motor.ChangeDutyCycle(base_speed - pid_out)
    right_motor.ChangeDutyCycle(base_speed + pid_out)


4. АНАЛОГ Random Walk (поиск объекта)
───────────────────────────────────────
  Roblox:  RandomWalker — W + случайные A/D + повороты мыши
  Робот:   ехать вперёд + плавные повороты моторов:
    left_motor.ChangeDutyCycle(60)
    right_motor.ChangeDutyCycle(75)          # лёгкий правый поворот
    time.sleep(random.uniform(0.5, 2.0))


5. ANTI-STUCK ДЛЯ РОБОТА
─────────────────────────
  Roblox: dist_history + optical_flow
  Робот:
    # Энкодеры колёс:
    ticks_1 = read_encoder()
    time.sleep(1.0)
    ticks_2 = read_encoder()
    if abs(ticks_2 - ticks_1) < MIN_TICKS: escape_maneuver()

    # IMU (MPU-6050):
    from mpu6050 import mpu6050
    accel = mpu6050(0x68).get_accel_data()
    if all(abs(a) < 0.05 for a in accel.values()): escape_maneuver()


6. СХЕМА ЖЕЛЕЗА (минимум)
──────────────────────────
  Raspberry Pi 5
    ├── Pi Camera v3 (CSI)     → 60fps, нет задержек USB
    ├── GPIO 12,13 → L298N     → левый мотор (PWM)
    ├── GPIO 18,19 → L298N     → правый мотор (PWM)
    ├── I2C → MPU-6050          → IMU anti-stuck
    └── USB → HC-05 Bluetooth   → дистанционный стоп


7. ИТОГОВОЕ СРАВНЕНИЕ
──────────────────────
  ┌────────────────────┬─────────────────────┬───────────────────────┐
  │ Задача             │ Roblox-бот          │ Реальный робот        │
  ├────────────────────┼─────────────────────┼───────────────────────┤
  │ Камера/кадр        │ mss/dxcam           │ cv2.VideoCapture      │
  │ Детекция           │ RT-DETR+ByteTrack   │ RT-DETR+ByteTrack (то же!) │
  │ Ошибка→сигнал      │ PID → mouse_event   │ PID → motor PWM       │
  │ Движение вперёд    │ KeyController W     │ GPIO PWM / Serial     │
  │ Поиск              │ RandomWalker        │ random drive loop     │
  │ Anti-stuck         │ dist+optical flow   │ encoder / IMU         │
  └────────────────────┴─────────────────────┴───────────────────────┘
═══════════════════════════════════════════════════════════════════════════════
"""
