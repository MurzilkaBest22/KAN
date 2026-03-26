ОПИСАНИЕ
KAN – библиотека для создания и обучения нейронных сетей Колмогорова‑Арнольда (KAN) на основе автоматического дифференцирования. Реализованы три режима обучения:
Forward Mode AD (прямой проход с дуальными числами)
Reverse Mode AD (классический обратный проход)
HyperDualReverse Mode AD (авторский метод на основе гипердуальных комбинаций)
Библиотека предназначена для демонстрации возможностей автоматического дифференцирования в машинном обучении. В текущей версии в качестве обучаемых функций используются полиномы. В будущем планируется расширение на другие элементарные функции.

ВОЗМОЖНОСТИ
Шаблонные классы LearnableFunction, Polynomial (можно легко расширить другими функциями).
Слой KAN_Layer.
Полная нейросеть KAN.
Функции обучения для каждого режима: Step_Forward, Step_Reverse, Step_HyperDualReverse.
Функции оценки ошибки.
Поддержка произвольного количества слоёв и нейронов.

УСТАНОВКА И СБОРКА
Библиотека является header‑only. Для использования достаточно скопировать папку KAN в ваш проект и подключить нужные заголовки:

#include "KAN/KAN_network.h"

Все классы находятся в пространстве имён kan.

ЗАВИСИМОСТИ
RealUtils
Rational (требуется для Dual)
Dual (основная библиотека автоматического дифференцирования)

ПРИМЕР (АППРОКСИМАЦИЯ СИНУСА)

#include <iostream>
#include <vector>
#include <cmath>
#include "KAN/KAN_network.h"

int main() {
    // Network: 1 input, 5 hidden neurons, 1 output; learning rate 0.01; polynomial degree 3
    kan::KAN<kan::Polynomial> net({1, 5, 1}, 0.01L, 3);
    std::cout << "Total parameters: " << net.totalParams() << std::endl;

    // Generate training data (sine on [0, 2π])
    const int examples = 100;
    std::vector<long double> X(examples), Y(examples);
    for (int i = 0; i < examples; ++i) {
        long double x = 2.0L * M_PI * i / examples;
        X[i] = (x - M_PI) / M_PI;   // normalize
        Y[i] = std::sin(x);
    }

    // Training
    const int steps = 500;
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < examples; ++i) {
            net.Step_Reverse({X[i]}, {Y[i]});
        }
        if (step % 50 == 0) {
            std::cout << "Step " << step << " loss: " << net.loss_view({X[0]}, {Y[0]}) << std::endl;
        }
    }

    // Testing
    for (int i = 0; i <= 20; ++i) {
        long double x = 2.0L * M_PI * i / 20;
        long double x_norm = (x - M_PI) / M_PI;
        auto out = net.forward_pass_Forward({x_norm});
        std::cout << "x=" << x << " sin=" << std::sin(x) << " pred=" << out[0].a << std::endl;
    }
    return 0;
}

ЛИЦЕНЗИЯ
MIT License. См. файл LICENSE.

АВТОР
Сычёв Михаил Денисович
Email: murzilkabest@icloud.com
Telegram: @Murz1k22
