"""
Murder Mystery 2 Bot Launcher
Лицензия: MIT (YOLOv10) или Apache 2.0 (RT-DETR)
"""

import sys
import os
import argparse
from pathlib import Path

# Добавляем путь к модулям
BASE_DIR = Path(__file__).parent
# Для standalone пакета все модули в той же директории
sys.path.insert(0, str(BASE_DIR))

def main():
    parser = argparse.ArgumentParser(
        description='Murder Mystery 2 Coin/Ball Collector Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mm2_bot_launcher.py                     # YOLOv10 (default)
  python mm2_bot_launcher.py --model rtdetr      # RT-DETR
  python mm2_bot_launcher.py --conf 0.3          # Custom confidence
  python mm2_bot_launcher.py --model rtdetr --conf 0.35
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolov10', 'rtdetr'],
        default='yolov10',
        help='Model to use (default: yolov10)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--no-window',
        action='store_true',
        help='Run without display window'
    )
    
    args = parser.parse_args()
    
    # Определяем пути к весам
    if args.model == 'yolov10':
        weights_path = BASE_DIR / 'weights' / 'ball_v10.pt'
        print("[INFO] Using YOLOv10 model (MIT License)")
        print("[INFO] Can be used commercially without restrictions")
    else:
        weights_path = BASE_DIR / 'weights' / 'ball_rtdetr.pt'
        print("[INFO] Using RT-DETR model (Apache 2.0 License)")
        print("[INFO] Can be used commercially without restrictions")
    
    # Проверка существования весов
    if not weights_path.exists():
        print(f"[ERROR] Model not found: {weights_path}")
        print("\nPlease ensure the model file is in the 'weights' folder.")
        print(f"Expected path: {weights_path}")
        input("Press Enter to exit...")
        return
    
    print(f"[OK] Model found: {weights_path}")
    print(f"[INFO] Confidence threshold: {args.conf}")
    print(f"[INFO] Show window: {not args.no_window}")
    print("\n" + "="*60)
    
    # Импортируем и запускаем соответствующий бот
    try:
        if args.model == 'yolov10':
            from run_yolov10_bot import YOLOv10Bot
            bot = YOLOv10Bot(
                weights_path=str(weights_path),
                conf_thres=args.conf,
                show_window=not args.no_window
            )
        else:
            from run_rtdetr_bot import RTDETRBot
            bot = RTDETRBot(
                weights_path=str(weights_path),
                conf_thres=args.conf,
                show_window=not args.no_window
            )
        
        # Запуск бота
        print("\n[START] Bot is running...")
        print("[INFO] Press 'Q' in the window to stop")
        print("[INFO] Or press Ctrl+C in console")
        print("="*60 + "\n")
        
        bot.run()
        
    except KeyboardInterrupt:
        print("\n[STOP] Bot stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == '__main__':
    main()

