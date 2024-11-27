import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os


# Функція для обробки одного зображення
def process_image(image_path, threshold=0.5):
    # Завантаження зображення
    image = Image.open(image_path).convert("RGB")

    # Перетворення в тензор і додавання batch dimension
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Прогноз
    with torch.no_grad():
        outputs = model(image_tensor)

    # Отримання результатів
    scores = outputs[0]["scores"].cpu().numpy()
    masks = outputs[0]["masks"].cpu().numpy()
    boxes = outputs[0]["boxes"].cpu().numpy()

    # Фільтрація за порогом впевненості
    indices = [i for i, score in enumerate(scores) if score > threshold]
    filtered_masks = masks[indices]
    filtered_scores = scores[indices]

    # Вивід scores у консоль
    print(f"Processing: {image_path}")
    for i, score in enumerate(filtered_scores):
        print(f"Object {i + 1}: Score = {score:.2f}")


    black_background = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)

    # Накладення знайдених об'єктів
    for i, mask in enumerate(filtered_masks):
        binary_mask = mask[0] > 0.5
        black_background[binary_mask] = np.array(image)[binary_mask]

        # Додавання тексту (процент збігу) на зображення
        font = ImageFont.load_default()
        annotated_image = Image.fromarray(black_background)
        draw = ImageDraw.Draw(annotated_image)
        text = f"{filtered_scores[i] * 100:.1f}%"
        draw.text((10, annotated_image.height - 30 * (i + 1)), text, fill=(255, 255, 255), font=font)
        black_background = np.array(annotated_image)

    return image, black_background


# Завантаження моделі
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Використання GPU, якщо доступний
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# arrImg
image_paths = ["img/000000000191.jpg", "img/000000000694.jpg", "img/000000000774.jpg", "img/000000000970.jpg", "img/000000002219.jpg", "img/000000002271.jpg", "img/000000002769.jpg"]  # Замінити на реальні шляхи

# Обробка кожного зображення
for image_path in image_paths:
    original_image, processed_image = process_image(image_path)

    # Візуалізація результату
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(processed_image)
    axs[1].set_title("Identification Objects")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()
