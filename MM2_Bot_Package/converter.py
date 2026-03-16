from ultralytics import YOLO

# Укажите путь к вашей обученной модели
model_path = 'C:/Users/user/Desktop/создание нейронки/MM2_Bot_Package/weights/candies_v10.pt'

print(f"Загрузка модели из {model_path}...")

# Загружаем вашу модель
model = YOLO(model_path)

print("Начинаю конвертацию в ONNX с FP16 (half-precision)...")

# Экспортируем модель
# format='onnx' -> создаст файл в формате ONNX
# half=True -> применит квантизацию до FP16
model.export(format='onnx', half=True)

print("\nГотово! ✨")
print(f"Рядом с вашей моделью должен был появиться файл 'peoples_yolov10m.onnx'.")
print("Это ваша новая, быстрая модель.")