#include "KAN_network.h"
#include "real_utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <exception>

void sin_test_Forward() {
	try {
		// Поставим в настройках нейросети 1 входной параметр, 5 скрытых и 1 выходной. Для аппроксимации ограничимся многочленами третьей степени
		kan::KAN<kan::Polynomial> Network({ 1, 5, 1 }, 0.01L, 3);

		std::cout << "Total parameters: " << Network.totalParams() << std::endl;

		const int examples = 100; // Генерация обучающих примеров в виде 100 точек на отрезке от 0 до 2 pi и значений синуса от них
		std::vector<long double> X(examples); // Входные значения
		std::vector<long double> Y(examples); // Выходные значения

		for (int i = 0; i < examples; ++i) {
			// Т. к. std::acos(-1) равен числу Пи
			long double x = (2.0L * ::pi() * i) / examples;
			X[i] = (x - ::pi()) / ::pi();
			Y[i] = std::sin(x);
		}

		const int steps = 500; // Количество шагов в обучении

		auto start = std::chrono::high_resolution_clock::now(); // Начало отсчёта времени на обучение

		for (int step = 0; step < steps; ++step) {
			long double summary_loss = 0.0;

			// Обучение
			for (int i = 0; i < examples; ++i) {
				Network.Step_Forward({ X[i] }, { Y[i] });
				summary_loss += Network.loss_view({ X[i] }, { Y[i] });
			}

			// Вывод средней ошибки каждые 20 шагов
			if (step % 20 == 0) {
				std::cout << "Step " << step << ", average loss: " << summary_loss / examples << std::endl;
			}
		}

		auto end = std::chrono::high_resolution_clock::now(); // Конец отсчёта времени
		std::chrono::duration<long double> time = end - start;

		std::cout << "Total training time: " << time.count() << " s\n"; // Вывод времени выполнения
		std::cout << "Average per step: " << time.count() / steps << " s\n"; // Среднее время выполнения одного шага

		// Проверка точности предсказаний нейросети
		std::cout << std::endl;
		for (int i = 0; i <= 20; ++i) {
			long double x = (2.0L * std::acos(-1.0L) * i) / 20;
			long double x_norm = (x - pi()) / pi();
			auto output_vector = Network.forward_pass_Forward({ x_norm }); // Вектор выходных значений
			long double predicted = output_vector[0].a; // Предсказание нейросети
			long double expected = std::sin(static_cast<long double> (x)); // Ожидаемый ответ
			std::cout << std::fixed << std::setprecision(4) << "x = " << x << ", sin(x) = " << expected << ", predicted = " << predicted << ", loss = " << std::abs(predicted - expected) << std::endl; // Вывод результатов
		}
	}
	catch (const std::exception& error) {
		std::cerr << "\n sin_test_Forward::Exception: " << error.what() << std::endl;
		throw;
	}
	catch (...) {
		std::cerr << "\n sin_test_Forward::Unknown exception" << std::endl;
		throw;
	}
}

void sin_test_Reverse() {
	try {
		// Поставим в настройках нейросети 1 входной параметр, 5 скрытых и 1 выходной. Для аппроксимации ограничимся многочленами третьей степени
		kan::KAN<kan::Polynomial> Network({ 1, 5, 1 }, 0.01L, 3);

		std::cout << "Total parameters: " << Network.totalParams() << std::endl;

		const int examples = 100; // Генерация обучающих примеров в виде 100 точек на отрезке от 0 до 2 pi и значений синуса от них
		std::vector<long double> X(examples); // Входные значения
		std::vector<long double> Y(examples); // Выходные значения

		for (int i = 0; i < examples; ++i) {
			// Т. к. std::acos(-1) равен числу Пи
			long double x = (2.0L * ::pi() * i) / examples;
			X[i] = (x - ::pi()) / ::pi();
			Y[i] = std::sin(x);
		}

		const int steps = 500; // Количество шагов в обучении

		auto start = std::chrono::high_resolution_clock::now(); // Начало отсчёта времени на обучение

		for (int step = 0; step < steps; ++step) {
			long double summary_loss = 0.0;

			// Обучение
			for (int i = 0; i < examples; ++i) {
				Network.Step_Reverse({ X[i] }, { Y[i] });
				summary_loss += Network.loss_view({ X[i] }, { Y[i] });
			}

			// Вывод средней ошибки каждые 20 шагов
			if (step % 20 == 0) {
				std::cout << "Step " << step << ", average loss: " << summary_loss / examples << std::endl;
			}
		}

		auto end = std::chrono::high_resolution_clock::now(); // Конец отсчёта времени
		std::chrono::duration<long double> time = end - start;

		std::cout << "Total training time: " << time.count() << " s\n"; // Вывод времени выполнения
		std::cout << "Average per step: " << time.count() / steps << " s\n"; // Среднее время выполнения одного шага

		// Проверка точности предсказаний нейросети
		std::cout << std::endl;
		for (int i = 0; i <= 20; ++i) {
			long double x = (2.0L * std::acos(-1.0L) * i) / 20;
			long double x_norm = (x - pi()) / pi();
			auto output_vector = Network.forward_pass_Forward({ x_norm }); // Вектор выходных значений
			long double predicted = output_vector[0].a; // Предсказание нейросети
			long double expected = std::sin(static_cast<long double> (x)); // Ожидаемый ответ
			std::cout << std::fixed << std::setprecision(4) << "x = " << x << ", sin(x) = " << expected << ", predicted = " << predicted << ", loss = " << std::abs(predicted - expected) << std::endl; // Вывод результатов
		}
	}
	catch (const std::exception& error) {
		std::cerr << "\n sin_test_Reverse::Exception: " << error.what() << std::endl;
		throw;
	}
	catch (...) {
		std::cerr << "\n sin_test_Reverse::Unknown exception" << std::endl;
		throw;
	}
}

void sin_test_HyperDualReverse() {
	try {
		// Поставим в настройках нейросети 1 входной параметр, 5 скрытых и 1 выходной. Для аппроксимации ограничимся многочленами третьей степени
		kan::KAN<kan::Polynomial> Network({ 1, 5, 1 }, 0.01L, 3);

		std::cout << "Total parameters: " << Network.totalParams() << std::endl;

		const int examples = 100; // Генерация обучающих примеров в виде 100 точек на отрезке от 0 до 2 pi и значений синуса от них
		std::vector<long double> X(examples); // Входные значения
		std::vector<long double> Y(examples); // Выходные значения

		for (int i = 0; i < examples; ++i) {
			// Т. к. std::acos(-1) равен числу Пи
			long double x = (2.0L * ::pi() * i) / examples;
			X[i] = (x - ::pi()) / ::pi();
			Y[i] = std::sin(x);
		}

		const int steps = 500; // Количество шагов в обучении

		auto start = std::chrono::high_resolution_clock::now(); // Начало отсчёта времени на обучение

		for (int step = 0; step < steps; ++step) {
			long double summary_loss = 0.0;

			// Обучение
			for (int i = 0; i < examples; ++i) {
				Network.Step_HyperDualReverse({ X[i] }, { Y[i] });
				summary_loss += Network.loss_view({ X[i] }, { Y[i] });
			}

			// Вывод средней ошибки каждые 20 шагов
			if (step % 20 == 0) {
				std::cout << "Step " << step << ", average loss: " << summary_loss / examples << std::endl;
			}
		}

		auto end = std::chrono::high_resolution_clock::now(); // Конец отсчёта времени
		std::chrono::duration<long double> time = end - start;

		std::cout << "Total training time: " << time.count() << " s\n"; // Вывод времени выполнения
		std::cout << "Average per step: " << time.count() / steps << " s\n"; // Среднее время выполнения одного шага

		// Проверка точности предсказаний нейросети
		std::cout << std::endl;
		for (int i = 0; i <= 20; ++i) {
			long double x = (2.0L * std::acos(-1.0L) * i) / 20;
			long double x_norm = (x - pi()) / pi();
			auto output_vector = Network.forward_pass_Forward({ x_norm }); // Вектор выходных значений
			long double predicted = output_vector[0].a; // Предсказание нейросети
			long double expected = std::sin(static_cast<long double> (x)); // Ожидаемый ответ
			std::cout << std::fixed << std::setprecision(4) << "x = " << x << ", sin(x) = " << expected << ", predicted = " << predicted << ", loss = " << std::abs(predicted - expected) << std::endl; // Вывод результатов
		}
	}
	catch (const std::exception& error) {
		std::cerr << "\n sin_test_HyperDualReverse::Exception: " << error.what() << std::endl;
		throw;
	}
	catch (...) {
		std::cerr << "\n sin_test_HyperDualReverse::Unknown exception" << std::endl;
		throw;
	}
}

int main() {
	std::cout << "SIN_TEST_FORWARD STARTS" << std::endl << std::endl;
	try {
		sin_test_Forward();
	}
	catch (const std::exception& error) {
		std::cerr << "\n Main: exception in sin_test_Forward: " << error.what() << std::endl;
		throw;
	}
	catch (...) {
		std::cerr << "\n Main: unknown exception in sin_test_Forward" << std::endl;
		throw;
	}

	std::cout << std::endl << std::endl;

	std::cout << "SIN_TEST_REVERSE STARTS" << std::endl << std::endl;
	try {
		sin_test_Reverse();
	}
	catch (const std::exception& error) {
		std::cerr << "\n Main: exception in sin_test_Reverse: " << error.what() << std::endl;
		throw;
	}
	catch (...) {
		std::cerr << "\n Main: unknown exception in sin_test_Reverse" << std::endl;
		throw;
	}

	std::cout << std::endl << std::endl;

	std::cout << "SIN_TEST_HYPERDUALREVERSE STARTS" << std::endl << std::endl;
	try {
		sin_test_HyperDualReverse();
	}
	catch (const std::exception& error) {
		std::cerr << "\n Main: exception in sin_test_HyperDualReverse: " << error.what() << std::endl;
		return 1;
	}
	catch (...) {
		std::cerr << "\n Main: unknown exception in sin_test_HyperDualReverse" << std::endl;
		return 1;
	}

	return 0;
}