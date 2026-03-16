"""
Скрипт для записи геймплея: захватывает окно Roblox и записывает видео + нажатия клавиш.
Данные сохраняются в папку gameplay_data/ для последующего обучения модели навигации.
"""
import cv2
import numpy as np
import keyboard
import time
import os
import json
from datetime import datetime
from threading import Event
import sys

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roblox.screen import getHandleByTitle, captureWindowMSS

# Конфигурация
RECORD_FPS = 30  # Частота записи видео
WINDOW_TITLE = "Roblox"  # Заголовок окна игры
STOP_HOTKEY = 'ctrl+shift+q'  # Горячая клавиша для остановки записи
DATA_DIR = "gameplay_data"
VIDEO_CODEC = 'mp4v'  # Кодек для записи видео

# Состояние записи
recording = Event()
recording.set()  # Начинаем запись

# Текущие нажатые клавиши (множество)
pressed_keys = set()

def key_event(e):
    """Обработчик событий клавиатуры."""
    global pressed_keys
    if e.event_type == keyboard.KEY_DOWN:
        pressed_keys.add(e.name)
    elif e.event_type == keyboard.KEY_UP:
        pressed_keys.discard(e.name)

def record_session(session_dir, window_handle):
    """Основная функция записи сеанса - записывает видео + клавиши."""
    # Создаем подкаталоги для видео и меток
    video_path = os.path.join(session_dir, 'gameplay.mp4')
    labels_dir = os.path.join(session_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Захватываем первый кадр для получения размера
    sample_frame = captureWindowMSS(window_handle, convert='GBR', save=None)
    height, width = sample_frame.shape[:2]
    
    # Инициализируем VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    out = cv2.VideoWriter(video_path, fourcc, RECORD_FPS, (width, height))
    
    if not out.isOpened():
        print(f"[ERROR] Не удалось создать видеофайл: {video_path}")
        return

    frame_count = 0
    start_time = time.time()

    print(f"[INFO] Начало записи в {session_dir}")
    print(f"[INFO] Частота записи: {RECORD_FPS} FPS, разрешение: {width}x{height}")
    print(f"[INFO] Для остановки нажмите {STOP_HOTKEY}")

    while recording.is_set():
        # Захват окна
        img = captureWindowMSS(window_handle, convert='GBR', save=None)
        
        # Запись кадра в видео
        out.write(img)
        
        # Сохранение меток (нажатые клавиши) для каждого кадра
        label = list(pressed_keys)
        label_filename = f"frame_{frame_count:06d}.json"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            json.dump({
                'keys': label,
                'timestamp': time.time() - start_time
            }, f)

        frame_count += 1

        # Вывод прогресса каждые 100 кадров
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] Записано {frame_count} кадров за {elapsed:.1f} сек.")

    out.release()
    print(f"[INFO] Запись остановлена. Всего кадров: {frame_count}, видео сохранено: {video_path}")

def stop_recording():
    """Остановка записи по горячей клавише."""
    global recording
    recording.clear()
    print("\n[INFO] Получен сигнал остановки. Завершение...")

def main():
    # Создаем корневую папку для данных, если её нет
    os.makedirs(DATA_DIR, exist_ok=True)

    # Получаем handle окна Roblox
    try:
        window_handle = getHandleByTitle(WINDOW_TITLE)
        print(f"[INFO] Окно '{WINDOW_TITLE}' найдено.")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("Убедитесь, что игра запущена и окно видимо.")
        return

    # Создаем папку для текущего сеанса с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(DATA_DIR, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    # Сохраняем конфигурацию сеанса
    config = {
        'fps': RECORD_FPS,
        'window_title': WINDOW_TITLE,
        'stop_hotkey': STOP_HOTKEY,
        'timestamp': timestamp,
        'codec': VIDEO_CODEC
    }
    with open(os.path.join(session_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Регистрируем обработчики клавиатуры
    keyboard.hook(key_event)
    keyboard.add_hotkey(STOP_HOTKEY, stop_recording)

    # Запускаем запись
    try:
        record_session(session_dir, window_handle)
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем.")
    finally:
        recording.clear()
        keyboard.unhook_all()
        print("[INFO] Запись завершена.")

if __name__ == "__main__":
    main()
