#pragma once // Защита от множественного включения

#include "KAN_layer.h"
#include "alldiff.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace kan {

	// Нейросеть KAN в виде шаблонного класса (Function - тип функции, совпадающий с используемым в слоях)
	template<typename Function>
	class KAN {
	private:
		std::vector<std::unique_ptr<KAN_Layer<Function>>> layers; // Вектор указателей на слои (объекты класса KAN_Layer<Function>)
		long double v; // Скорость обучения
		int all_parameters; // Общее количество параметров во всей нейронной сети

	public:
		// Конструктор
		template<typename ... Arguments>
		KAN(const std::vector<int>& layers_sizes, long double V, Arguments&&... Args) : v(V), all_parameters(0) {
			if (layers_sizes.size() < 2) {
				throw std::invalid_argument("KAN: at least two layers required");
			}

			// Последовательное создание слоёв
			for (size_t i = 0; i < layers_sizes.size() - 1; ++i) {
				int input_Dim = layers_sizes[i];
				int output_Dim = layers_sizes[i + 1];

				// Создание слоя KAN_Layer<Function> с передачей ему размерности и аргументов Args...
				auto layer = std::make_unique<KAN_Layer<Function>>(input_Dim, output_Dim, std::forward<Arguments>(Args)...);

				// Добавление слоя в вектор слоёв
				layers.push_back(std::move(layer));
			}

			// Присваивание глобальных индексов параметрам всех слоёв
			int next_index = 0;
			for (auto& layer : layers) {
				next_index = layer->Global_Index(next_index);
			}
			all_parameters = next_index; // Последний индекс будет равен общему количеству параметров
		}

		// Деструктор по умолчанию
		~KAN() = default;

		// Запрет копирования
		KAN(const KAN&) = delete;
		KAN& operator=(const KAN&) = delete;

		// Разрешение перемещения
		KAN(KAN&&) = default;
		KAN& operator=(KAN&&) = default;

		// Чтение общего количества параметров сети
		int totalParams() const { return all_parameters; }

		// Прямой проход при вещественных параметрах
		std::vector<dual::Dual> forward_pass_Forward(const std::vector<long double>& input) const {
			// Замена вектора вещественных входных значений на дуальные
			std::vector<dual::Dual> input_dual(input.size());

			for (size_t i = 0; i < input.size(); ++i) {
				input_dual[i] = dual::Dual(input[i], 0.0);
			}

			// Последовательный проход всех слоёв
			for (const auto& layer : layers) {
				// отрицательные значение индекса в методе forward_pass_Forward() класса KAN_Layer означает отсутствие параметров, к которым прибавляется нильпотентный элемент
				input_dual = layer->forward_pass_Forward(input_dual, -1);
			}

			return input_dual;
		}

		// Прямой проход при прибавлении нильпотентного элемента к параметру с индексом k
		std::vector<dual::Dual> forward_pass_Forward(const std::vector<long double>& input, int k) const {
			// Замена вектора вещественных входных значений на дуальные
			std::vector<dual::Dual> input_dual(input.size());

			for (size_t i = 0; i < input.size(); ++i) {
				input_dual[i] = dual::Dual(input[i], 0.0);
			}

			// Последовательный проход всех слоёв
			for (const auto& layer : layers) {
				input_dual = layer->forward_pass_Forward(input_dual, k);
			}

			return input_dual;
		}

		// Среднеквадратичная ошибка при прибавлении нильпотентного элемента к параметру с глобальным индексом k
		dual::Dual MSE(const std::vector<long double>& input, const std::vector<long double>& expected_output, int k) const {
			auto calculated_output = forward_pass_Forward(input, k);

			if (calculated_output.size() != expected_output.size()) {
				throw std::invalid_argument("KAN::MSE: expected_output size mismatch");
			}

			dual::Dual loss(0.0, 0.0);

			for (size_t i = 0; i < expected_output.size(); ++i) {
				auto difference = calculated_output[i] - expected_output[i];
				loss = loss + difference * difference;
			}

			return loss;
		}

		// Один шаг обучения на одном примере
		void Step_Forward(const std::vector<long double>& input, const std::vector<long double>& expected_output) {
			// Градиент
			std::vector<long double> grad(all_parameters, 0.0);

			// Вычисление функции потерь по каждому параметру
			for (int i = 0; i < all_parameters; ++i) {
				dual::Dual loss = MSE(input, expected_output, i);
				grad[i] = loss.b; // Так как нильпотентная часть ошибки при добавлении к одному параметру нильпотентого элемента соответствует производной функции потерь по этому параметру
			}

			// Обновление параметров с проходом по слоям и передачей им соответствующих частей градиента
			int b = 0;
			for (auto& layer : layers) {
				layer->GradUpdate(grad, v, b);
			}
		}

		long double loss_view(const std::vector<long double>& input, const std::vector<long double>& expected_output) const {
			return MSE(input, expected_output, -1).a;
		}

		// Прямой проход для Reverse Mode AD
		std::vector<long double> forward_pass_Reverse(const std::vector<long double>& input) const {
			std::vector<long double> current = input;
			for (const auto& layer : layers) {
				current = layer->forward_pass_Reverse(current);
			}
			return current;
		}

		// Вычисление градиентов с помощью Reverse Mode AD
		std::vector<long double> grad_Reverse(const std::vector<long double>& input, const std::vector<long double>& expected_output) {
			auto calculated_output = forward_pass_Reverse(input);
			if (calculated_output.size() != expected_output.size()) {
				throw std::invalid_argument("KAN::grad_Reverse: expected_output size mismatch");
			}

			// Градиент среднеквадратичной ошибки на выходе
			std::vector<long double> gradient_output(calculated_output.size());
			for (size_t i = 0; i < calculated_output.size(); ++i) {
				gradient_output[i] = 2.0 * (calculated_output[i] - expected_output[i]);
			}

			// Вектор градиентов параметров
			std::vector<long double> gradient_parameters(all_parameters, 0.0L);

			// Обратный проход от последнего слоя к первому
			std::vector<long double> gradient_input = gradient_output;
			for (auto i = layers.rbegin(); i != layers.rend(); ++i) {
				auto& layer = *i;
				std::vector<long double> new_gradient_input(layer->get_input_dim(), 0.0L);
				layer->backward_pass_Reverse(gradient_input, gradient_parameters, new_gradient_input);
				gradient_input = std::move(new_gradient_input);
			}

			// Возвращение вычисленного полного градиента
			return gradient_parameters;
		}

		// Шаг с помощью Reverse Mode AD
		void Step_Reverse(const std::vector<long double>& input, const std::vector<long double>& expected_output) {
			// Получение градиента за один проход
			auto grad = grad_Reverse(input, expected_output);

			// Обновление параметров с проходом по слоям и передачей им соответствующих частей градиента
			int b = 0;
			for (auto& layer : layers) {
				layer->GradUpdate(grad, v, b);
			}
		}

		// HyperDualReverse-вариант прямого прохода
		std::vector<dual::HyperDualCombination> forward_pass_HyperDualReverse(const std::vector<dual::HyperDualCombination>& input) const {
			if (input.size() == 0) {
				throw std::invalid_argument("KAN::forward_pass_HyperDualReverse: input size must be > 0");
			}
			std::vector<dual::HyperDualCombination> current = input;
			// Последовательный проход всех слоёв
			for (const auto& layer : layers) {
				const auto& orders = input[0].get_orders();
				current = layer->forward_pass_HyperDualReverse(current, orders);
			}

			return current;
		}

		// Среднеквадратичная ошибка в HyperDualReverse Mode AD
		dual::HyperDualCombination MSE_HyperDualReverse(const std::vector<long double>& input, const std::vector<long double>& expected_output) const {
			// Порядки гипердуальной комбинации должны быть равны 2 для градиента первого порядка
			std::vector<size_t> orders(all_parameters, 2);

			std::vector<dual::HyperDualCombination> input_HyperDual(input.size());
			for (size_t i = 0; i < input.size(); ++i) {
				input_HyperDual[i] = dual::HyperDualCombination(orders, input[i]);
			}

			auto calculated_output_HyperDual = forward_pass_HyperDualReverse(input_HyperDual);

			if (calculated_output_HyperDual.size() != expected_output.size()) {
				throw std::invalid_argument("KAN::MSE_HyperDualReverse: expected_output size mismatch");
			}

			dual::HyperDualCombination loss_HyperDual(orders);
			for (size_t i = 0; i < calculated_output_HyperDual.size(); ++i) {
				dual::HyperDualCombination difference = calculated_output_HyperDual[i] - expected_output[i];
				loss_HyperDual = loss_HyperDual + difference * difference;
			}

			return loss_HyperDual;
		}

		// Вычисление градиентов с помощью HyperDualReverse Mode AD
		std::vector<long double> grad_HyperDualReverse(const std::vector<long double>& input, const std::vector<long double>& expected_output) {
			std::vector<long double> gradient(all_parameters, 0.0L);
			dual::HyperDualCombination MSE_HyperDual = MSE_HyperDualReverse(input, expected_output);
			for (int j = 0; j < all_parameters; ++j) {
				std::vector<size_t> multi_index(all_parameters, 0);
				multi_index[j] = 1;
				size_t linear_index = MSE_HyperDual.linear_index(multi_index);
				gradient[j] = MSE_HyperDual[linear_index];
			}

			return gradient;
		}

		// Шаг с помощью HyperDualReverse Mode AD
		void Step_HyperDualReverse(const std::vector<long double>& input, const std::vector<long double>& expected_output) {
			// Получение градиента за один проход
			auto grad = grad_HyperDualReverse(input, expected_output);

			// Обновление параметров с проходом по слоям и передачей им соответствующих частей градиента
			int b = 0;
			for (auto& layer : layers) {
				layer->GradUpdate(grad, v, b);
			}
		}

		long double HyperDualReverse_loss_view(const std::vector<long double>& input, const std::vector<long double>& expected_output) const {
			return MSE_HyperDualReverse(input, expected_output)[0];
		}
	};
}