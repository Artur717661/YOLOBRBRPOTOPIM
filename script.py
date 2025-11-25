import os
import cv2
from ultralytics import YOLO
from collections import Counter


def run_food_detection():

    model_path = 'runs/detect/food_result/weights/best.pt'

    image_path = 'test.jpg'


    price_list = {
        "apple": 30,
        "cake": 120,
        "tea": 25,
        "soup": 90,
        "cutlet": 85,
        "puree": 50,
        "salad": 70,

    }


    if not os.path.exists(model_path):
        print(f" Ошибка: Не найден файл модели: {model_path}")
        print("Проверь, закончилось ли обучение и правильно ли указан путь.")
        return

    if not os.path.exists(image_path):
        print(f" Ошибка: Не найдена картинка {image_path}")
        print("Положи фото еды в папку проекта и назови его test.jpg")
        return


    print(f"Загружаем модель из {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    print("Анализируем фото...")

    results = model.predict(source=image_path, conf=0.25, save=True)

    result = results[0]
    detected_items = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        item_name = model.names[class_id]
        detected_items.append(item_name)

    print("\n" + "=" * 30)
    print(f"{'РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ':^30}")
    print("=" * 30)

    if not detected_items:
        print(" Еда не обнаружена.")
    else:
        total_price = 0
        counts = Counter(detected_items)

        for item, count in counts.items():
            price = price_list.get(item, 0)
            sum_price = price * count
            total_price += sum_price

            display_name = item