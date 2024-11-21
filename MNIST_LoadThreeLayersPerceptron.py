# Імпортуємо необхідні бібліотеки
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Визначаємо параметри
input_size = 28 * 28  # Розмір входу (28x28 пікселів для зображень MNIST)
hidden_size = 128      # Кількість нейронів у прихованих шарах
num_classes = 10       # Кількість класів (цифри від 0 до 9)
batch_size = 1         # Розмір батчу

# Використовуємо трансформації для нормалізації даних
transform = transforms.Compose([
    transforms.ToTensor(),  # Перетворення зображень у тензори
    transforms.Normalize((0.5,), (0.5,))  # Нормалізація (середнє 0.5, стандартне відхилення 0.5)
])

# Визначаємо модель три перцептрону
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Перший шар
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Другий шар
        self.fc3 = nn.Linear(hidden_size, num_classes) # Третій шар

    def forward(self, x):
        x = x.view(-1, input_size)  # Перетворення зображення в вектор
        x = torch.relu(self.fc1(x))  # Активація ReLU для прихованого шару
        x = torch.relu(self.fc2(x)) # Активація ReLU для прихованого шару
        x = self.fc3(x)  # Прямий прохід через вихідний шар
        return x

# Функція для тестування моделі
def test_model():
    model.eval()  # Переключаємо модель в режим тестування
    with torch.no_grad():  # Вимикаємо обчислення градієнтів
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)  # Прогноз
            _, predicted = torch.max(outputs.data, 1)  # Отримуємо класи з прогнозів
            total += labels.size(0)  # Загальна кількість зображень
            correct += (predicted == labels).sum().item()  # Кількість правильних прогнозів
        accuracy = 100 * correct / total  # Обчислюємо точність
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')


# Завантажуємо датасет MNIST
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# Створюємо DataLoader для тестових даних
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Вказуємо шлях до файлу з моделлю
model_path = 'models/MNIST_ThreeLayersPerceptron.pth'
# Ініціалізуємо модель
model = SimpleNN()
# Завантажуємо збережені параметри
model.load_state_dict(torch.load(model_path))

# Тестуємо модель
test_model()