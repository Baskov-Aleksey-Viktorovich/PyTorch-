# Імпортуємо необхідні бібліотеки
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Визначаємо параметри
input_size = 28 * 28  # Розмір входу (28x28 пікселів для зображень MNIST)
hidden_size = 128      # Кількість нейронів у прихованому шарі
num_classes = 10       # Кількість класів (цифри від 0 до 9)
num_epochs = 5         # Кількість епох для навчання
batch_size = 64        # Розмір батчу
learning_rate = 0.001  # Швидкість навчання

# Використовуємо трансформації для нормалізації даних
transform = transforms.Compose([
    transforms.ToTensor(),  # Перетворення зображень у тензори
    transforms.Normalize((0.5,), (0.5,))  # Нормалізація (середнє 0.5, стандартне відхилення 0.5)
])

# Завантажуємо датасет MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Створюємо DataLoader для навчальних та тестових даних
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Визначаємо модель двошарового перцептрону
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Перший шар
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Другий шар
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Третій шар

    def forward(self, x):
        x = x.view(-1, input_size)  # Перетворення зображення в вектор
        x = torch.relu(self.fc1(x))  # Активація ReLU для прихованого шару
        x = self.fc2(x)  # Прямий прохід через вихідний шар
        x = self.fc3(x)  # Прямий прохід через вихідний шар
        return x

# Ініціалізуємо модель, втрачену функцію та оптимізатор
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Втрата для класифікації
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Оптимізатор Adam

# Функція для навчання моделі
def train_model():
    model.train()  # Переключаємо модель в режим навчання
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()  # Обнуляємо градієнти
            outputs = model(images)  # Прогноз
            loss = criterion(outputs, labels)  # Обчислюємо втрату
            loss.backward()  # Обчислюємо градієнти
            optimizer.step()  # Оновлюємо параметри моделі
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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

# Навчаємо модель
train_model()

# Тестуємо модель
test_model()