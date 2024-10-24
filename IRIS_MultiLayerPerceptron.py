import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Завантаження та підготовка даних
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Стандартизація даних
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Розділення на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Перетворення у тензори
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# 2. Створення моделі
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Вхідний шар (4 особливості) -> 1-й прихований шар (10 нейронів)
        self.fc2 = nn.Linear(10, 10)  # 1-й прихований шар -> 2-й прихований шар (10 нейронів)
        self.fc3 = nn.Linear(10, 3)   # 2-й прихований шар -> вихідний шар (3 класи)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()

# 3. Визначення функції втрат та оптимізатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Тренування моделі
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Обнулити градієнти
    outputs = model(X_train_tensor)  # Прогонка вперед
    loss = criterion(outputs, y_train_tensor)  # Обчислення втрат
    loss.backward()  # Обчислення градієнтів
    optimizer.step()  # Оновлення ваг

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Тестування моделі
model.eval()  # Переведення в режим оцінювання
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)  # Вибір класу з найбільшою ймовірністю
    accuracy = accuracy_score(y_test, predicted.numpy())
    print(f'Accuracy: {accuracy * 100:.2f}%')