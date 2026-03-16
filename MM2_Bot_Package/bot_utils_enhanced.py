"""
Утилиты для улучшенного бота поиска мячиков
Содержит вспомогательные функции и классы
"""

import numpy as np
import cv2
import time
from collections import deque
import json

class TargetTracker:
    """Трекер целей для отслеживания движения объектов"""
    
    def __init__(self, max_history=10):
        self.target_history = deque(maxlen=max_history)
        self.current_targets = {}
        self.target_id_counter = 0
    
    def update_targets(self, detections, frame_time):
        """Обновляет трекинг целей"""
        # Простая система трекинга на основе расстояния
        for detection in detections:
            cx, cy = detection['cx'], detection['cy']
            
            # Ищем ближайший существующий трек
            closest_track_id = None
            min_distance = float('inf')
            
            for track_id, track_data in self.current_targets.items():
                last_pos = track_data['positions'][-1]
                distance = np.sqrt((cx - last_pos['cx'])**2 + (cy - last_pos['cy'])**2)
                
                if distance < min_distance and distance < 50:  # Порог для ассоциации
                    min_distance = distance
                    closest_track_id = track_id
            
            if closest_track_id is not None:
                # Обновляем существующий трек
                self.current_targets[closest_track_id]['positions'].append({
                    'cx': cx, 'cy': cy, 'conf': detection['conf'], 'time': frame_time
                })
                self.current_targets[closest_track_id]['last_seen'] = frame_time
            else:
                # Создаем новый трек
                track_id = self.target_id_counter
                self.target_id_counter += 1
                self.current_targets[track_id] = {
                    'positions': [{'cx': cx, 'cy': cy, 'conf': detection['conf'], 'time': frame_time}],
                    'created': frame_time,
                    'last_seen': frame_time
                }
        
        # Удаляем старые треки
        current_time = time.time()
        tracks_to_remove = []
        for track_id, track_data in self.current_targets.items():
            if current_time - track_data['last_seen'] > 2.0:  # 2 секунды без обновления
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.current_targets[track_id]
    
    def get_stable_targets(self, min_observations=3):
        """Возвращает стабильные цели (наблюдаемые несколько раз)"""
        stable_targets = []
        for track_id, track_data in self.current_targets.items():
            if len(track_data['positions']) >= min_observations:
                # Вычисляем среднюю позицию и уверенность
                positions = track_data['positions']
                avg_cx = np.mean([p['cx'] for p in positions])
                avg_cy = np.mean([p['cy'] for p in positions])
                avg_conf = np.mean([p['conf'] for p in positions])
                
                stable_targets.append({
                    'track_id': track_id,
                    'cx': avg_cx,
                    'cy': avg_cy,
                    'conf': avg_conf,
                    'observations': len(positions),
                    'stability': len(positions) / 10.0  # Нормализованная стабильность
                })
        
        return stable_targets

class PerformanceMonitor:
    """Монитор производительности бота"""
    
    def __init__(self):
        self.fps_history = deque(maxlen=100)
        self.detection_times = deque(maxlen=50)
        self.movement_times = deque(maxlen=50)
        self.start_time = time.time()
    
    def log_fps(self, fps):
        """Логирует FPS"""
        self.fps_history.append(fps)
    
    def log_detection_time(self, detection_time):
        """Логирует время детекции"""
        self.detection_times.append(detection_time)
    
    def log_movement_time(self, movement_time):
        """Логирует время обработки движения"""
        self.movement_times.append(movement_time)
    
    def get_stats(self):
        """Возвращает статистику производительности"""
        stats = {
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'min_fps': np.min(self.fps_history) if self.fps_history else 0,
            'max_fps': np.max(self.fps_history) if self.fps_history else 0,
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_movement_time': np.mean(self.movement_times) if self.movement_times else 0,
            'runtime': time.time() - self.start_time
        }
        return stats

class SmartSearchEngine:
    """Умный движок поиска с адаптивными паттернами"""
    
    def __init__(self):
        self.search_patterns = [
            # Паттерн 1: Медленный поворот
            {'type': 'slow_turn', 'direction': 'right', 'duration': 0.4, 'pause': 0.2},
            {'type': 'slow_turn', 'direction': 'left', 'duration': 0.3, 'pause': 0.15},
            
            # Паттерн 2: Быстрый сканирование
            {'type': 'quick_scan', 'direction': 'right', 'duration': 0.15, 'pause': 0.05},
            {'type': 'quick_scan', 'direction': 'left', 'duration': 0.15, 'pause': 0.05},
            {'type': 'quick_scan', 'direction': 'right', 'duration': 0.15, 'pause': 0.05},
            
            # Паттерн 3: Движение + поиск
            {'type': 'move_search', 'direction': 'forward', 'duration': 0.3, 'pause': 0.1},
            {'type': 'move_search', 'direction': 'turn_right', 'duration': 0.2, 'pause': 0.1},
            {'type': 'move_search', 'direction': 'forward', 'duration': 0.3, 'pause': 0.1},
            {'type': 'move_search', 'direction': 'turn_left', 'duration': 0.2, 'pause': 0.1},
        ]
        
        self.current_pattern = 0
        self.pattern_success_rate = {}
        self.last_detection_time = 0
    
    def get_next_pattern(self):
        """Возвращает следующий паттерн поиска"""
        pattern = self.search_patterns[self.current_pattern % len(self.search_patterns)]
        self.current_pattern += 1
        return pattern
    
    def update_success_rate(self, pattern_type, success):
        """Обновляет статистику успешности паттернов"""
        if pattern_type not in self.pattern_success_rate:
            self.pattern_success_rate[pattern_type] = {'success': 0, 'total': 0}
        
        self.pattern_success_rate[pattern_type]['total'] += 1
        if success:
            self.pattern_success_rate[pattern_type]['success'] += 1
    
    def get_best_patterns(self):
        """Возвращает наиболее успешные паттерны"""
        if not self.pattern_success_rate:
            return self.search_patterns[:3]  # Возвращаем первые 3 по умолчанию
        
        # Сортируем по успешности
        sorted_patterns = sorted(
            self.pattern_success_rate.items(),
            key=lambda x: x[1]['success'] / max(1, x[1]['total']),
            reverse=True
        )
        
        # Возвращаем топ-3 паттерна
        best_types = [p[0] for p in sorted_patterns[:3]]
        return [p for p in self.search_patterns if p['type'] in best_types]

class AdaptiveController:
    """Адаптивный контроллер для настройки параметров бота"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=20)
        self.detection_history = deque(maxlen=30)
        self.movement_history = deque(maxlen=30)
        
        # Параметры для адаптации
        self.base_params = {
            'conf_threshold': 0.25,
            'min_distance': 50,
            'turn_threshold': 25,
            'search_threshold': 8,
            'jump_interval': 12
        }
        
        self.current_params = self.base_params.copy()
    
    def update_performance(self, fps, detections_count, collections_count):
        """Обновляет данные о производительности"""
        self.performance_history.append({
            'fps': fps,
            'detections': detections_count,
            'collections': collections_count,
            'time': time.time()
        })
    
    def adapt_parameters(self):
        """Адаптирует параметры на основе производительности"""
        if len(self.performance_history) < 10:
            return  # Недостаточно данных для адаптации
        
        recent_performance = list(self.performance_history)[-10:]
        avg_fps = np.mean([p['fps'] for p in recent_performance])
        avg_detections = np.mean([p['detections'] for p in recent_performance])
        avg_collections = np.mean([p['collections'] for p in recent_performance])
        
        # Адаптация порога уверенности
        if avg_fps < 5.0:  # Низкий FPS - повышаем порог
            self.current_params['conf_threshold'] = min(0.4, 
                self.current_params['conf_threshold'] + 0.02)
        elif avg_fps > 8.0 and avg_detections < 2:  # Высокий FPS, мало детекций - снижаем порог
            self.current_params['conf_threshold'] = max(0.15, 
                self.current_params['conf_threshold'] - 0.01)
        
        # Адаптация расстояния сбора
        if avg_collections / max(1, avg_detections) < 0.3:  # Низкая эффективность сбора
            self.current_params['min_distance'] = max(30, 
                self.current_params['min_distance'] - 5)
        elif avg_collections / max(1, avg_detections) > 0.8:  # Высокая эффективность
            self.current_params['min_distance'] = min(80, 
                self.current_params['min_distance'] + 3)
        
        # Адаптация порога поиска
        if avg_detections < 1:  # Мало детекций - начинаем поиск раньше
            self.current_params['search_threshold'] = max(5, 
                self.current_params['search_threshold'] - 1)
        elif avg_detections > 5:  # Много детекций - можем ждать дольше
            self.current_params['search_threshold'] = min(15, 
                self.current_params['search_threshold'] + 1)
    
    def get_current_params(self):
        """Возвращает текущие параметры"""
        return self.current_params.copy()

def calculate_target_priority(cx, cy, conf, distance, frame_shape, target_history=None):
    """Расширенный расчет приоритета цели"""
    frame_h, frame_w = frame_shape[:2]
    center_x = frame_w / 2
    center_y = frame_h / 2
    
    # Базовый приоритет
    priority = conf * 100
    
    # Бонус за близость к центру
    center_distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    center_bonus = max(0, 60 - center_distance / 8)
    priority += center_bonus
    
    # Бонус за близость к игроку
    distance_bonus = max(0, 120 - distance / 4)
    priority += distance_bonus
    
    # Штрафы за неудобные позиции
    if cy < frame_h * 0.25:  # Слишком высоко
        priority *= 0.6
    elif cy > frame_h * 0.8:  # Слишком низко
        priority *= 0.8
    
    if cx < frame_w * 0.1 or cx > frame_w * 0.9:  # По краям
        priority *= 0.7
    
    # Бонус за стабильность (если есть история)
    if target_history:
        stability_bonus = calculate_stability_bonus(cx, cy, target_history)
        priority += stability_bonus
    
    return priority

def calculate_stability_bonus(cx, cy, target_history):
    """Вычисляет бонус за стабильность цели"""
    if len(target_history) < 2:
        return 0
    
    recent_positions = list(target_history)[-5:]  # Последние 5 позиций
    if len(recent_positions) < 2:
        return 0
    
    # Вычисляем дисперсию позиций
    positions_x = [pos['cx'] for pos in recent_positions]
    positions_y = [pos['cy'] for pos in recent_positions]
    
    variance_x = np.var(positions_x)
    variance_y = np.var(positions_y)
    
    # Чем меньше дисперсия, тем больше бонус
    stability = 1.0 / (1.0 + variance_x + variance_y)
    return stability * 30  # Максимальный бонус 30

def optimize_detection_params(model, test_image, device):
    """Оптимизирует параметры детекции для лучшей производительности"""
    test_sizes = [320, 416, 512, 640]
    best_size = 416
    best_fps = 0
    
    for size in test_sizes:
        start_time = time.time()
        for _ in range(10):  # Тестируем 10 раз
            results = model(test_image, imgsz=size, device=device, verbose=False)
        end_time = time.time()
        
        fps = 10 / (end_time - start_time)
        if fps > best_fps:
            best_fps = fps
            best_size = size
    
    print(f"[OPTIMIZE] Лучший размер: {best_size}, FPS: {best_fps:.1f}")
    return best_size

def save_performance_report(stats, filename="performance_report.json"):
    """Сохраняет отчет о производительности"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'stats': stats,
        'summary': {
            'total_runtime': stats.get('runtime', 0),
            'avg_fps': stats.get('avg_fps', 0),
            'detection_efficiency': stats.get('avg_detection_time', 0),
            'movement_efficiency': stats.get('avg_movement_time', 0)
        }
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[REPORT] Отчет сохранен в {filename}")
    except Exception as e:
        print(f"[ERROR] Ошибка сохранения отчета: {e}")
