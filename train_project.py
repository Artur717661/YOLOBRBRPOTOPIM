import os
import yaml
from ultralytics import YOLO

def train_yolo_automatically():

    dataset_dir = os.path.abspath('SmartCanteen')
    yaml_path = os.path.join(dataset_dir, 'data.yaml')


    if not os.path.exists(yaml_path):
        print(f"ОШИБКА: Файл {yaml_path} не найден!")
        print("Убедитесь, что скрипт лежит РЯДОМ с папкой SmartCanteen.")
        return

    print(f"Найден датасет: {dataset_dir}")

    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)


    data_config['path'] = dataset_dir
    data_config['train'] = 'train/images'
    data_config['val'] = 'valid/images' 
    data_config['test'] = 'test/images'

    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print("Конфигурация data.yaml успешно обновлена.")

    model = YOLO('yolov8n.pt') 

    print("Запуск обучения...")
    results = model.train(
        data=yaml_path,   
        epochs=50,        
        imgsz=640,        
        batch=16,         
        name='smart_canteen_run', 
        device='cpu'      # Поставь 0, если у тебя есть видеокарта NVIDIA
    )
    
    print(f"Обучение завершено! Лучшая модель сохранена в: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_yolo_automatically()
