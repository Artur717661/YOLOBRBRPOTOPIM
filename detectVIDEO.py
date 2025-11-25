import os
import cv2
from ultralytics import YOLO
from collections import Counter


def run_food_detection():
    model_path = 'runs/detect/food_result/weights/best.pt'

    if not os.path.exists(model_path):
        print(f" Ошибка: Не найден файл модели: {model_path}")
        print("Проверь, закончилось ли обучение и правильно ли указан путь.")
        return

    print(f"Загружаем модель из {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Ошибка: Не удалось подключиться к веб-камере!")
        return

    print("Запуск веб-камеры... Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Ошибка: Не удалось получить кадр с камеры!")
            break

        results = model.predict(source=frame, conf=0.25, verbose=False)
        result = results[0]
        detected_items = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            item_name = model.names[class_id]
            detected_items.append(item_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{item_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        counts = Counter(detected_items)

        y_offset = 30
        cv2.putText(frame, "ОБНАРУЖЕННЫЕ ПРОДУКТЫ:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for item, count in counts.items():
            y_offset += 25
            cv2.putText(frame, f"{item}: {count}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_count = len(detected_items)
        cv2.putText(frame, f"ВСЕГО ОБЪЕКТОВ: {total_count}",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Food Detection - Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Веб-камера отключена.")


if __name__ == "__main__":
    run_food_detection()