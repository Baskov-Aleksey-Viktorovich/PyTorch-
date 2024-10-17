import numpy as np  # Імпортуємо бібліотеку NumPy для роботи з масивами
from fontTools.ttLib.tables.S__i_l_f import pass_attrs_fsm


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Функція активації: у нашому випадку це буде сигмоїда
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Обчислюємо значення сигмоїди

# Похідна сигмоїдної функції для використання в зворотному поширенні помилки
def sigmoid_derivative(prediction):
    return prediction * (1 - prediction)  # Обчислюємо похідну сигмоїди

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Клас для одношарового перцептрона
class SingleLayerPerceptron:

    def __init__(self, input_size, learning_rate = 0.1, init_value = 'random', save_train_history = False):

        self.learning_rate = learning_rate                              # Визначаємо швидкість навчання

        if init_value == 'random':
            self.weights = np.random.rand(input_size)                   # Ініціалізуємо ваги випадковими значеннями
            self.bias = np.random.rand(1)                               # Ініціалізуємо зсув випадковим значення
        elif type(init_value) in [float, int]:
            self.bias = init_value                                      # Ініціалізуємо зсув значенням
            self.weights = np.ones(input_size) * init_value             # Ініціалізуємо ваги значенням
        else:
            self.bias = 0                                               # Ініціалізуємо зсув нулем
            self.weights = np.zeros(input_size)                         # Ініціалізуємо ваги нулями

        self.save_train_history = save_train_history

        if bool(self.save_train_history):
            self.trajectory = []
        else:
            self.trajectory = None

        return

    # Метод для прямого проходження (forward pass)
    def predict(self, inputs):

        weighted_sum = np.dot(inputs, self.weights) + self.bias         # Обчислюємо зважену суму

        return sigmoid(weighted_sum)                                    # Повертаємо результат активації

    # Метод для навчання перцептрона
    def train(self, inputs, expected_output, epochs = 10000, bRepeat = False):

        for epoch in range(epochs):                                     # Цикл для кожної епохи
            for x, y in zip(inputs, expected_output):                   # Проходимо через кожен зразок даних
                prediction = self.predict(x)                            # Прогнозуємо вихід
                error = y - prediction                                  # Обчислюємо помилку

                # Оновлюємо ваги та зсув
                self.weights += self.learning_rate * error * sigmoid_derivative(prediction) * x
                self.bias += self.learning_rate * error * sigmoid_derivative(prediction)

            if bool(self.save_train_history):
                self.trajectory.append((np.array(self.bias), np.array(self.weights)))

        return