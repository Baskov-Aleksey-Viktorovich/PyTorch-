import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Параметри
batch_size = 64  # Розмір партії
learning_rate = 0.001  # Швидкість навчання
num_epochs = 5  # Кількість епох

# Преобразування даних
transform = transforms.Compose([
    transforms.ToTensor(),  # Перетворення зображень у тензори
    transforms.Normalize((0.5,), (0.5,))  # Нормалізація
])

# Завантаження навчального набору даних
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Завантаження тестового набору даних
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Визначення моделі
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Перший сверточний шар
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Другий сверточний шар
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Перший повнозв'язний шар
        self.fc2 = nn.Linear(128, 10)  # Другий повнозв'язний шар (для 10 класів)
        self.pool = nn.MaxPool2d(2, 2)  # Шар підвибірки
        self.relu = nn.ReLU()  # Активаційна функція ReLU

    def forward(self, x):
        # Прямий прохід через мережу
        x = self.pool(self.relu(self.conv1(x)))  # Перший сверточний шар з підвибіркою
        x = self.pool(self.relu(self.conv2(x)))  # Другий сверточний шар з підвибіркою
        x = x.view(-1, 64 * 7 * 7)  # Перетворення тензора для повнозв'язного шару
        x = self.relu(self.fc1(x))  # Перший повнозв'язний шар
        x = self.fc2(x)  # Другий повнозв'язний шар
        return x

# Ініціалізація моделі, функції втрат та оптимізатора
model = CNN()  # Створення екземпляра моделі
criterion = nn.CrossEntropyLoss()  # Функція втрат
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Оптимізатор Adam

# Навчання моделі
for epoch in range(num_epochs):
    model.train()  # Перехід в режим навчання
    for images, labels in train_loader:
        optimizer.zero_grad()  # Обнулення градієнтів
        outputs = model(images)  # Прогон через модель
        loss = criterion(outputs, labels)  # Обчислення втрат
        loss.backward()  # Зворотне поширення градієнтів
        optimizer.step()  # Оновлення параметрів

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Виведення втрат

# Оцінка моделі
model.eval()  # Перехід в режим оцінки
correct = 0
total = 0
with torch.no_grad():  # Без обчислення градієнтів
    for images, labels in test_loader:
        outputs = model(images)  # Прогон через модель
        _, predicted = torch.max(outputs.data, 1)  # Отримання передбачень
        total += labels.size(0)  # Загальна кількість зображень
        correct += (predicted == labels).sum().item()  # Підрахунок правильних відповідей

print(f'Accuracy: {100 * correct / total:.2f}%')  # Виведення точності

#---------------------------------------------------------------------------

# # Зберігаємо модель у *.onnx файл
dummy_input = torch.randn(1, 1, 28, 28) # Підготовка тестового вхідного тензора

torch.onnx.export(  # Експорт
    model,
    dummy_input,
    r"mnist-load.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)