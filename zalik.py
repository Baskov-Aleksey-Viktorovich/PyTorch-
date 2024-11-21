import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Генерація даних для варіанту B
X = np.linspace(0, 25, 100)
#Y = X**3 / 125 - 2 * X**2 - 10
Y = np.cos(X) * X / (X + 1) + np.random.normal(0, 0.05, size=X.shape)  # реальні дані з шумом для варіанту B
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

# Перетворення на тензори
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Визначення нейромережі
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# Ініціалізація моделі, функції втрат і оптимізатора
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Навчання моделі
num_epochs = 5000
for epoch in range(num_epochs):
    # Прямий прохід
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    
    # Зворотний прохід і оптимізація
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Друк втрат кожні 100 епох
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Прогноз
predicted = model(X_tensor).detach().numpy()

# Візуалізація результатів
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Реальні дані')  # Відображаємо реальні дані
plt.plot(X, predicted, color='red', label='Передбачення моделі', linewidth=2)  # Прогнозовані дані
plt.legend()  # Додаємо легенду
plt.xlabel('X')  # Підпис осі X
plt.ylabel('Y')  # Підпис осі Y
plt.title('Нелінійна регресія за допомогою нейронної мережі (Варіант B)')  # Заголовок графіка
plt.show()  # Виводимо графік
