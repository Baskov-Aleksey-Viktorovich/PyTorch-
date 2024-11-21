# Імпортуємо необхідні бібліотеки
import torch  # для роботи з тензорами і обчисленнями
from torchvision import transforms  # для перетворень зображень перед передачею в модель
from torchvision.models.detection import maskrcnn_resnet50_fpn  # імпортуємо модель Mask R-CNN на базі ResNet50
import matplotlib.pyplot as plt  # для візуалізації результатів
from PIL import Image  # для роботи з зображеннями
import numpy as np

# 1. Завантажуємо попередньо натреновану модель Mask R-CNN
# Використовуємо модель maskrcnn_resnet50_fpn, яка побудована на базі ResNet50 та використовує FPN (Feature Pyramid Network)
model = maskrcnn_resnet50_fpn(pretrained=True)

# 2. Переводимо модель в режим оцінки (evaluation mode)
# У цьому режимі модель не буде використовувати Dropout і BatchNorm працюватиме в режимі тестування
model.eval()

# 3. Визначаємо пристрій (CPU чи GPU)
# Якщо є доступ до GPU, ми використовуємо його, в іншому випадку модель працюватиме на CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Переміщаємо модель на обраний пристрій
model.to(device)

# 4. Попередня обробка зображення
# Трансформації для перетворення зображення в тензор
transform = transforms.Compose([
    transforms.ToTensor(),  # Перетворюємо зображення в тензор (відмінності у значеннях пікселів)
])

# 5. Завантажуємо зображення, яке потрібно сегментувати
# Вказуємо шлях до зображення

#image_path = r"data\coco2017\test2017\000000000904.jpg"  # Шлях до зображення
image_path = r"data\coco2017\test2017\000000003903.jpg"  # Шлях до зображення
#image_path = r"data\ScreenShotImage.jpg"
#image_path = r"data\ImageForTest_01.JPG"
image = Image.open(image_path)  # Завантажуємо зображення за допомогою PIL

# 6. Перетворюємо зображення в тензор та додаємо додаткову розмірність (batch size = 1)
# Модель чекає на вхід тензор з формою [batch_size, 3, H, W], тому додаємо додаткову вимірність
image_tensor = transform(image).unsqueeze(0)  # unsqueeze(0) додає розмір партії (batch) рівний 1

# 7. Переміщаємо тензор зображення на той самий пристрій, що й модель (GPU або CPU)
image_tensor = image_tensor.to(device)

# 8. Проганяємо модель через зображення для отримання прогнозів
# Використовуємо torch.no_grad(), оскільки ми не тренуємо модель, а лише робимо передбачення
with torch.no_grad():
    prediction = model(image_tensor)  # Отримуємо передбачення: рамки, маски, класи, ймовірності

# 9. Отримуємо маски, мітки класів та ймовірності з результатів
boxes = prediction[0]['boxes'] # рамки сегментації (тензор розміру [num_instances, 1, H, W])
masks = prediction[0]['masks']  # маски сегментації (тензор розміру [num_instances, 1, H, W])
labels = prediction[0]['labels']  # мітки класів, що виявлені на зображенні
scores = prediction[0]['scores']  # ймовірності для кожної маски (представляють упевненість моделі)

print('Boxes:', boxes)
print('Labels:', labels)
print('Scores:', scores)

#-----------------------------------------------------------------------------------------------------------------------
# 10. Функція для візуалізації результатів сегментації
# Вона відображає оригінальне та масковане зображення
def show_results(image, masks, labels, scores, prob_threshold = 0.5, mask_level = 0.5):
    image_mask = np.array(image)
    mask = np.zeros(masks[0,0].shape)
    for i in range(masks.shape[0]):  # ітеруємо через всі виявлені об'єкти
        # Якщо ймовірність для даної маски більша за поріг беремо її до уваги
        if scores[i] > prob_threshold:
            mask[:,:] = np.where(masks[i,0] > mask_level, 1, mask[:,:])

    image_mask[:, :, 0] = mask[:, :] * image_mask[:, :, 0]
    image_mask[:, :, 1] = mask[:, :] * image_mask[:, :, 1]
    image_mask[:, :, 2] = mask[:, :] * image_mask[:, :, 2]

    # Відображаємо оригінальне зображення та лише зображення по масці
    plt.figure(1)
    plt.imshow(image)
    plt.figure(2)
    plt.imshow(image_mask)
    # Показуємо результат
    plt.show()

# 11. Викликаємо функцію для візуалізації результатів сегментації
# Покажемо маски з ймовірністю більше 0.5 (можна змінити поріг за необхідності)
show_results(image, masks, labels, scores, 0.2, 0.5)
