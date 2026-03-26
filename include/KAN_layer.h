#pragma once // Защита от множественного включения

#include "KAN_function.h"
#include "alldiff.h"
#include <vector>
#include <memory>
#include <type_traits>
#include <utility>

namespace kan {

	template<typename Function>
	class KAN_Layer {
		// Проверка наследуемости между Function и LearnableFunction
		static_assert(std::is_base_of<LearnableFunction, Function>::value, "Function must inherit from LearnableFunction");

	private:
		int input_dim; // Размерность вектора входных данных (количество входящих на слой переменных)
		int output_dim; // Размерность вектора выходных данных (количество выходящих из слоя переменных)

		// Матрица функций, в которой F[j][i] будет связывать вход i с выходом j
		std::vector<std::vector<std::unique_ptr<Function>>> f_matrix;
		int number_of_parameters;

		// Векторы последних значений для Reverse Mode AD
		mutable std::vector<long double> last_input;
		mutable std::vector<long double> last_output;

	public:
		// Конструктор для создания слоя с заданными размерностями и аргументами для конструктора Function
		template<typename... Arguments>
		KAN_Layer(int input_dimension, int output_dimension, Arguments&&... Args) : input_dim(input_dimension), output_dim(output_dimension), number_of_parameters(0) {
			f_matrix.resize(output_dim);

			for (int j = 0; j < output_dim; ++j) {
				f_matrix[j].resize(input_dim);

				for (int i = 0; i < input_dim; ++i) {
					// Создание функции с передачей в неё параметров
					f_matrix[j][i] = std::make_unique<Function>(std::forward<Arguments>(Args)...);
					// Увеличение общего количества параметров
					number_of_parameters += f_matrix[j][i]->numParams();
				}
			}
		}

		// Деструктор по умолчанию (автоматическое удаление всех функций входит в функционал unique_ptr)
		~KAN_Layer() = default;

		// Запрет копирования
		KAN_Layer(const KAN_Layer&) = delete;
		KAN_Layer& operator=(const KAN_Layer&) = delete;

		// Разрешение перемещения
		KAN_Layer(KAN_Layer&&) = default;
		KAN_Layer& operator=(KAN_Layer&&) = default;

		// Извлечение полей
		int get_input_dim() const { return input_dim; }
		int get_output_dim() const { return output_dim; }
		int get_number_of_parameters() const { return number_of_parameters; }

		int Global_Index(int start_index) {
			for (int j = 0; j < output_dim; ++j) {
				for (int i = 0; i < input_dim; ++i) {
					f_matrix[j][i]->setFirstParamIndex(start_index);
					start_index += f_matrix[j][i]->numParams();
				}
			}
			return start_index;
		}

		// Прямой проход слоя (k - индекс параметра, передаваемого в метод evaluate_Forward() каждой функции)
		std::vector<dual::Dual> forward_pass_Forward(const std::vector<dual::Dual>& input, int k) const {
			if (input.size() != static_cast<size_t>(input_dim)) {
				throw std::invalid_argument("KAN_Layer::forward_pass_Forward: input size mismatch");
			}

			std::vector<dual::Dual> output(output_dim, dual::Dual(0.0, 0.0));

			for (int j = 0; j < output_dim; ++j) {
				for (int i = 0; i < input_dim; ++i) {
					output[j] = output[j] + f_matrix[j][i]->evaluate_Forward(input[i], k);
				}
			}

			return output;
		}

		// Обновление параметров слоя по градиенту
		void GradUpdate(const std::vector<long double>& full_gradient, long double v, int& b) {
			for (int j = 0; j < output_dim; ++j) {
				for (int i = 0; i < input_dim; ++i) {
					int params = f_matrix[j][i]->numParams();
					std::vector<long double> f_grad(params);
					for (int k = 0; k < params; ++k) {
						f_grad[k] = full_gradient[b + k];
					}
					f_matrix[j][i]->update(f_grad, v);
					b += params;
				}
			}
		}

		// Прямой проход слоя с использованием Reverse Mode AD
		std::vector<long double> forward_pass_Reverse(const std::vector<long double>& input) const {
			if (input.size() != static_cast<size_t>(input_dim)) {
				throw std::invalid_argument("KAN_Layer::forward_pass_Reverse: input size mismatch");
			}

			last_input = input;

			std::vector<long double> output(output_dim, 0.0L);

			for (int j = 0; j < output_dim; ++j) {
				for (int i = 0; i < input_dim; ++i) {
					output[j] = output[j] + f_matrix[j][i]->forward_pass_Reverse(input[i]);
				}
			}

			last_output = output;
			return output;
		}
		
		// Обратный проход слоя с использованием Reverse Mode AD
		void backward_pass_Reverse(const std::vector<long double>& gradient_output, std::vector<long double>& gradient_parameters, std::vector<long double>& gradient_input) const {
			if (gradient_output.size() != static_cast<size_t>(output_dim)) {
				throw std::invalid_argument("KAN_Layer::backward_pass_Reverse: gradient_output size mismatch");
			}
			if (gradient_input.size() != static_cast<size_t>(input_dim)) {
				gradient_input.resize(input_dim, 0.0L);
			}

			// Обнуление gradient_input
			std::fill(gradient_input.begin(), gradient_input.end(), 0.0L);

			// Накопление градиентов при обратном проходе слоя
			for (int j = 0; j < output_dim; ++j) {
				long double gradient_out = gradient_output[j];
				for (int i = 0; i < input_dim; ++i) {
					auto& func = f_matrix[j][i];
					int params = func->numParams();
					std::vector<long double> f_gradient_parameters(params, 0.0L);
					long double f_gradient_input = 0.0L;
					func->backward_pass_Reverse(gradient_out, f_gradient_parameters, f_gradient_input);

					int start_index = func->firstParamIndex();
					for (int k = 0; k < params; ++k) {
						gradient_parameters[start_index + k] += f_gradient_parameters[k];
					}

					gradient_input[i] += f_gradient_input;
				}
			}

		}

		// Прямой проход слоя с использованием HyperDualReverse Mode AD
		std::vector<dual::HyperDualCombination> forward_pass_HyperDualReverse(const std::vector<dual::HyperDualCombination>& input, const std::vector<size_t>& orders) const {
			if (input.size() != static_cast<size_t>(input_dim)) {
				throw std::invalid_argument("KAN_Layer::forward_pass_HyperDualReverse: input size mismatch");
			}

			std::vector<dual::HyperDualCombination> output(output_dim, dual::HyperDualCombination(orders));

			for (int j = 0; j < output_dim; ++j) {
				for (int i = 0; i < input_dim; ++i) {
					int first_index = f_matrix[j][i]->firstParamIndex();
					output[j] = output[j] + f_matrix[j][i]->evaluate_HyperDualReverse(input[i], orders, first_index);
				}
			}

			return output;
		}
	};
}
