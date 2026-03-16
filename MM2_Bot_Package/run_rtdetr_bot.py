"""
RT-DETR Bot v8 Fixed: No GUI, no crashes, smooth PID, GPU auto
Core only: detect → PID → collect → search. FPS 5-7.
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import random  # FIXED: Import random

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from roblox.screen import CaptureStream
from roblox.control import Control
from roblox.utils import FrameCounter

class RTDETRBot:
    def __init__(self, weights_path, conf_thres=0.25):
        print(f"[INIT] Загрузка RT-DETR: {weights_path}")
        
        self.device = '0' if torch.cuda.is_available() else ''
        print(f"[INFO] Device: {'cuda:0' if self.device else 'CPU'}")
        
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        
        print("[INIT] Захват экрана...")
        self.stream = CaptureStream("Roblox", saveInterval=0)
        
        self.control = Control()
        self.frame_counter = FrameCounter()
        self.frame_counter.fps = 0.0
        
        # PID (улучшенные коэффициенты)
        self.pid_kp = 0.035
        self.pid_ki = 0.0015
        self.pid_kd = 0.025
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.05
        self.integral_limit = 100.0
        
        # Params (улучшенные)
        self.min_distance = 55
        self.turn_threshold = 25
        self.paste_delay = 0.25
        self.predict_size = 416  # Увеличено для лучшей детекции
        
        self.frames_without_coins = 0
        self.search_threshold = 5
        self.is_searching = False
        self.is_moving_to_coin = False
        self.frames_since_jump = 0
        self.jump_interval = 15
        self.previous_distance = float('inf')
        self.frames_after_collect = 0
        self.post_collect_frames = 10
        self.retry_frames = 0
        self.retry_threshold = 3
        
        print("[OK] Bot готов!")
        print(f"[INFO] Conf: {self.conf_thres}, Min dist: {self.min_distance}px")
        print("\n[START] Ctrl+C для остановки\n")
    
    def find_closest_coins(self, results, frame_shape, max_coins=2):
        frame_h, frame_w = frame_shape[:2]
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        coins = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                conf = boxes.conf[i].item()
                cls = int(boxes.cls[i].item())
                
                if cls == 0 and conf >= self.conf_thres:
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    norm_x = cx / frame_w
                    norm_y = cy / frame_h
                    
                    if norm_y < 0.3 or norm_x < 0.15 or norm_x > 0.85:
                        continue
                    
                    distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    score = distance * (1 - conf)
                    
                    coins.append({'cx': cx, 'cy': cy, 'conf': conf, 'distance': distance, 'score': score})
        
        coins.sort(key=lambda c: c['score'])
        return coins[:max_coins]
    
    def move_to_coin(self, coin, frame_shape):
        frame_w = frame_shape[1]
        center_x = frame_w / 2
        center_y = frame_shape[0] / 2
        
        cx = coin['cx']
        cy = coin['cy']
        distance = coin['distance']
        conf = coin['conf']
        
        current_distance = distance
        error = cx - center_x
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0
        pid_output = self.pid_kp * error + self.pid_ki * self.integral + self.pid_kd * derivative
        
        progress = self.previous_distance - current_distance > 10
        self.prev_error = error
        
        if self.is_searching:
            self.control.release_all_keys()
            self.is_searching = False
            print("[STOP SEARCH] Монета найдена!")
        
        if distance < self.min_distance:
            self.control.release_all_keys()
            time.sleep(self.paste_delay)
            
            print(f"[COLLECT] Подобрали! (conf: {conf:.2f}, dist: {distance:.1f}px)")
            self.integral = 0.0
            self.previous_distance = float('inf')
            self.is_moving_to_coin = False
            self.frames_after_collect = self.post_collect_frames
            return True
        
        turn_time = abs(pid_output) / 10
        turn_time = max(0.05, min(0.2, turn_time))
        if distance < 200:
            turn_time *= 0.5
        
        self.control.release_all_keys()
        
        if abs(error) > self.turn_threshold:
            if error < 0:
                self.control.press('left')
            else:
                self.control.press('right')
            self.control.press('up')
            time.sleep(float(turn_time))  # FIXED: Convert to float
        else:
            self.control.press('up')
            time.sleep(0.1)
        
        if not progress:
            self.retry_frames += 1
            if self.retry_frames > self.retry_threshold:
                print("[RETRY] Нет прогресса — retry!")
                self.control.release_all_keys()
                time.sleep(0.2)
                self.control.press('jump')
                time.sleep(0.3)
                self.control.press('up')
                self.retry_frames = 0
        else:
            self.retry_frames = 0
        
        self.previous_distance = current_distance
        self.is_moving_to_coin = True
        
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval and distance > 150:
            self.control.press('jump')
            time.sleep(0.1)
            self.frames_since_jump = 0
            print("[JUMP] Для приближения!")
        
        return False
    
    def start_search(self):
        if not self.is_searching:
            self.is_searching = True
            self.is_moving_to_coin = False
            print("[SEARCH] Поиск...")
        
        self.control.release_all_keys()
        
        direction = 'right' if self.frames_without_coins % 2 == 0 else 'left'
        self.control.press(f'turn_{direction}')
        time.sleep(0.2 + random.uniform(0, 0.1))  # FIXED: random imported
        self.control.release_all_keys()
        time.sleep(0.1)
        
        self.frames_since_jump += 1
        if self.frames_since_jump >= self.jump_interval:
            self.control.press('jump')
            time.sleep(0.1)
            self.frames_since_jump = 0
    
    def run(self):
        try:
            prev_time = time.time()
            for img, img0 in self.stream:
                results = self.model(img, imgsz=self.predict_size, device=self.device, conf=self.conf_thres, verbose=False)
                
                collected = False
                if len(results) > 0:
                    coins = self.find_closest_coins(results, img.shape, max_coins=2)
                    
                    if coins:
                        self.frames_without_coins = 0
                        
                        closest_coin = coins[0]
                        
                        collected = self.move_to_coin(closest_coin, img.shape)
                        
                        print(f"[DETECT] Conf: {closest_coin['conf']:.2f}, Dist: {closest_coin['distance']:.1f}px (top {len(coins)})")
                    else:
                        if self.frames_after_collect <= 0:
                            self.frames_without_coins += 1
                        else:
                            self.frames_after_collect -= 1
                else:
                    if self.frames_after_collect <= 0:
                        self.frames_without_coins += 1
                    else:
                        self.frames_after_collect -= 1
                
                if self.frames_without_coins >= self.search_threshold and self.frames_after_collect <= 0:
                    self.start_search()
                
                current_time = time.time()
                self.dt = current_time - prev_time
                prev_time = current_time
                self.frame_counter.log()
                
                # No GUI — clean run
        
        except KeyboardInterrupt:
            print("\n[STOP] Ctrl+C...")
        
        finally:
            print("[EXIT] Bot остановлен")

def main():
    parser = argparse.ArgumentParser(description='RT-DETR Bot v8 Final')
    parser.add_argument('--weights', type=str, default='weights/ball_rtdetr.pt', help='Путь к весам')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
    parser.add_argument('--no-window', action='store_true', help='Без окна')  # Ignored, no GUI
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"[ERROR] Веса не найдены: {args.weights}")
        return
    
    bot = RTDETRBot(args.weights, args.conf)
    bot.run()

if __name__ == '__main__':
    main()