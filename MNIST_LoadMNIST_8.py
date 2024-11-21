import onnx
import onnxruntime as ort
import numpy as np

# Вказуємо шлях до файлу з моделлю
model_path = 'mnist-8.onnx'

# Завантажте модель
model = onnx.load(model_path)

# Перевірте, чи модель валідна
onnx.checker.check_model(model)
print("Модель успішно завантажена і валідна.")

# Створіть сесію для виконання моделі
ort_session = ort.InferenceSession(model_path)

# Отримайте ім'я вхідного шару
input_name = ort_session.get_inputs()[0].name

# Створемо рандомний масив NumPy
input_data = np.random.randn(28, 28).astype(np.float32)

input_data[4:22,14] = 50
input_data[22,10:19] = 50

# Нормалізуйте дані (пікселі від 0 до 255, нормалізуємо до 0-1)
input_data = input_data / 255.0

# Додайте розмірність для пакування (1, 1, 28, 28) - (batch size, channels, height, width)
tensor_input_data = input_data.reshape(1, 1, 28, 28)

# Виконайте передбачення
output = ort_session.run(None, {input_name: tensor_input_data})

# Виведіть результат
predicted_class = np.argmax(output[0])
print("Передбачене значення:", predicted_class)

import matplotlib.pyplot as plt

plt.imshow(input_data, cmap='gray')
plt.title(f"Передбачене значення: {predicted_class}")
plt.show()