import numpy as np
import matplotlib.pyplot as plt
from classes_Perceptron import SingleLayerPerceptron as SLP

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Основна частина програми
if __name__ == "__main__":

    bLogicalAnd = False
    bLogicalOr = False
    bClassification = True

    if bLogicalAnd:
        #-------------------------------------------------------------------------------------------------------------------
        # Оператор логічного "I"
        #-------------------------------------------------------------------------------------------------------------------

        # Вхідні дані для логічного оператора "І"
        inputs_and = np.array \
            ([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ])

        # Очікувані виходи для логічного оператора "І"
        expected_output_and = np.array \
            ([
                [0],
                [0],
                [0],
                [1]
            ])

        # Ініціалізуємо перцептрон з 2 входами
        size_inputs = 2
        perceptron = SLP(input_size = size_inputs, save_train_history = True)

        # Тренуємо перцептрон на операторі "І"
        print("\nТренування на операторі 'І':")
        num_epochs = 10000
        perceptron.train(inputs_and, expected_output_and, epochs = num_epochs)
        print("Значення зсуву після тренування", perceptron.bias)
        print("Вагові значення після тренування",perceptron.weights)

        # Тестуємо модель на операторі "І"
        print("Тестування на операторі 'І':")
        for x in inputs_and:
            print(f"Вхід: {x}, Прогноз: {perceptron.predict(x)}")

        if bool(perceptron.trajectory):
            biases = []
            weights = []
            for indx in range(num_epochs):
                biases.append(perceptron.trajectory[indx][0][0])
                weights.append(perceptron.trajectory[indx][1])


            plt.figure(1)
            plt.plot([np.log10(abs(biases[indx] - biases[indx-1])) for indx in range(1, num_epochs)])

            for weight_index in range(size_inputs):
                plt.plot([np.log10(abs(weights[index][weight_index] - weights[index - 1][weight_index])) for index in range(1, num_epochs)])

        del(perceptron)
        if bool(perceptron.trajectory):
            del(biases)
            del(weights)

    if bLogicalOr:
        #-------------------------------------------------------------------------------------------------------------------
        # Оператор логічного "АБО"
        #-------------------------------------------------------------------------------------------------------------------

        # Вхідні дані для логічного оператора "АБО"
        inputs_or = np.array \
            ([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ])

        # Очікувані виходи для логічного оператора "АБО"
        expected_output_or = np.array \
            ([
                [0],
                [1],
                [1],
                [1]
            ])

        # Ініціалізуємо новий перцептрон для "АБО", оскільки ваги з попереднього навчання не підходять
        size_inputs = 2
        perceptron = SLP(input_size = size_inputs, save_train_history = True)

        # Тренуємо перцептрон на операторі "АБО"
        print("\nТренування на операторі 'АБО':")
        num_epochs = 10000
        perceptron.train(inputs_or, expected_output_or, epochs = num_epochs)
        print("Значення зсуву після тренування", perceptron.bias)
        print("Вагові значення після тренування",perceptron.weights)

        # Тестуємо модель на операторі "АБО"
        print("Тестування на операторі 'АБО':")
        for x in inputs_or:
            print(f"Вхід: {x}, Прогноз: {perceptron.predict(x)}")

        if bool(perceptron.trajectory):
            biases = []
            weights = []
            for indx in range(num_epochs):
                biases.append(perceptron.trajectory[indx][0][0])
                weights.append(perceptron.trajectory[indx][1])

            plt.figure(1)
            plt.plot([np.log10(abs(biases[indx] - biases[indx-1])) for indx in range(1, num_epochs)])

            for weight_index in range(size_inputs):
                plt.plot([np.log10(abs(weights[index][weight_index] - weights[index - 1][weight_index])) for index in range(1, num_epochs)])

            plt.show()

        del(perceptron)
        if bool(perceptron.trajectory):
            del (biases)
            del (weights)

    if bClassification:
        import os
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        #-------------------------------------------------------------------------------------------------------------------
        # Класифікація рукописних цифр
        #-------------------------------------------------------------------------------------------------------------------

        # Завантажуємо дані MNIST
        from keras.datasets import mnist
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        # Нормалізуємо дані для тренування
        inputs_train = np.zeros( (len(train_X), np.prod(train_X[0].shape)) )
        for index in range(len(train_X)):
            inputs_train[index,:] = np.array(train_X[index].flatten()/256.)

        expected_output_train = np.zeros(len(train_y))
        for index in range(len(train_y)):
            expected_output_train[index] = train_y[index]

        # Ініціюємо перцептрон
        size_inputs = inputs_train.shape[1]
        perceptron = SLP(input_size = size_inputs, save_train_history = False)

        # Тренуємо модель
        num_epochs = 1000
        perceptron.train(inputs_train, expected_output_train, epochs = num_epochs)

        if bool(perceptron.trajectory):
            biases = []
            weights = []
            for indx in range(num_epochs):
                biases.append(perceptron.trajectory[indx][0][0])
                weights.append(perceptron.trajectory[indx][1])

            plt.figure(1)
            plt.plot([(abs(biases[indx] - biases[indx-1])) for indx in range(1, num_epochs)])

            import random
            for weight_index in [random.choice(range(size_inputs)), random.choice(range(size_inputs))]:
                plt.plot([(abs(weights[index][weight_index] - weights[index - 1][weight_index])) for index in range(1, num_epochs)])

            plt.show()

        # Нормалізуємо дані для тестування
        inputs_test = np.zeros( (len(test_X), np.prod(test_X[0].shape)) )
        for index in range(len(test_X)):
            inputs_test[index,:] = np.array(test_X[index].flatten()/256.)

        expected_output_test = np.zeros(len(test_y))
        for index in range(len(test_y)):
            expected_output_test[index] = test_y[index]

        # Оцінюємо точність
        predictions = perceptron.predict(inputs_test)
        accuracy = np.mean(predictions == expected_output_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        # Видаляємо об'єкти
        del(perceptron)
        if bool(perceptron.trajectory):
            del(biases)
            del(weights)

    print('Done')