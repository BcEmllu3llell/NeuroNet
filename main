#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

// Функция активации ReLU
double relu(double x) {
    return (x > 0) ? x : 0;
}

// Производная функции активации ReLU
double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

// Функция прямого прохода (forward pass)
double forward(double input1, double input2, const vector<double>& weights1, const vector<double>& weights2, const vector<double>& bias1, const vector<double>& bias2) {
    // Прямой проход через первый слой
    double z1_1 = input1 * weights1[0] + bias1[0];
    double z1_2 = input2 * weights1[1] + bias1[1];
    double a1_1 = relu(z1_1);
    double a1_2 = relu(z1_2);

    // Прямой проход через второй слой
    double z2 = a1_1 * weights2[0] + a1_2 * weights2[1] + bias2[0];
    return relu(z2); // Финальный выход
}

// Функция для обучения сети (градиентный спуск)
void train(const vector<vector<double>>& inputs, const vector<double>& targets, vector<double>& weights1, vector<double>& weights2, vector<double>& bias1, vector<double>& bias2, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            double input1 = inputs[i][0];
            double input2 = inputs[i][1];
            double target = targets[i];

            // Прямой проход через нейронную сеть
            double z1_1 = input1 * weights1[0] + bias1[0];
            double z1_2 = input2 * weights1[1] + bias1[1];
            double a1_1 = relu(z1_1);
            double a1_2 = relu(z1_2);
            double z2 = a1_1 * weights2[0] + a1_2 * weights2[1] + bias2[0];
            double output = relu(z2);

            // Ошибка
            double error = target - output;

            // Вычисление градиентов для второго слоя
            double d_output = error * relu_derivative(z2);
            double d_a1_1 = d_output * weights2[0] * relu_derivative(z1_1);
            double d_a1_2 = d_output * weights2[1] * relu_derivative(z1_2);

            // Обновление весов второго слоя
            weights2[0] += learning_rate * d_output * a1_1;
            weights2[1] += learning_rate * d_output * a1_2;
            bias2[0] += learning_rate * d_output;

            // Обновление весов первого слоя
            weights1[0] += learning_rate * d_a1_1 * input1;
            weights1[1] += learning_rate * d_a1_2 * input2;
            bias1[0] += learning_rate * d_a1_1;
            bias1[1] += learning_rate * d_a1_2;
        }

        // Печать ошибки на каждой эпохе (опционально)
        if (epoch % 1000 == 0) {
            double total_error = 0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                total_error += pow(targets[i] - forward(inputs[i][0], inputs[i][1], weights1, weights2, bias1, bias2), 2);
            }
            cout << "Epoch " << epoch << ", Error: " << total_error << endl;
        }
    }
}

int main() {
    // Пример входных данных (XOR)
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> targets = {0, 1, 1, 0}; // Ожидаемые результаты для XOR

    // Инициализация весов и смещений случайными значениями
    vector<double> weights1 = {rand() / double(RAND_MAX), rand() / double(RAND_MAX)};
    vector<double> weights2 = {rand() / double(RAND_MAX), rand() / double(RAND_MAX)};
    vector<double> bias1 = {rand() / double(RAND_MAX), rand() / double(RAND_MAX)};
    vector<double> bias2 = {rand() / double(RAND_MAX)};

    // Обучаем сеть на 10000 эпохах с шагом обучения 0.1
    train(inputs, targets, weights1, weights2, bias1, bias2, 10000, 0.1);

    // Проверка после обучения
    cout << "Output for (0, 0): " << forward(0, 0, weights1, weights2, bias1, bias2) << endl;
    cout << "Output for (0, 1): " << forward(0, 1, weights1, weights2, bias1, bias2) << endl;
    cout << "Output for (1, 0): " << forward(1, 0, weights1, weights2, bias1, bias2) << endl;
    cout << "Output for (1, 1): " << forward(1, 1, weights1, weights2, bias1, bias2) << endl;

    return 0;
}
