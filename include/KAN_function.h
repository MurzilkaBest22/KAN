#pragma once // Защита от множественного включения

#include "alldiff.h"
#include <vector>
#include <random>
#include <stdexcept>

namespace kan {

	// Абстрактный (для этого используется virtual) класс для любой обучаемой функции одной переменной
	class LearnableFunction {
	public:
		virtual ~LearnableFunction() = default;

		// Подсчёт количества параметров функции
		virtual int numParams() const = 0;

		// Установка глобального индекса для первого параметра функции в общей структуре параметров всей нейросети для однозначной индентификации
		virtual void setFirstParamIndex(int first_index) = 0;

		// Возвращение глобального индекса первого параметра (т. е. в обратную сторону)
		virtual int firstParamIndex() const = 0;

		// Вычисление значения функции от дуального аргумента x, при котором к параметру с индексом j прибавляется нильпотентный элемент
		virtual dual::Dual evaluate_Forward(const dual::Dual& x, int j) const = 0;

		// Обновление параметров функции при помощи вектора градиентов (gradient) и заданной скорости обучения (v)
		virtual void update(const std::vector<long double>& gradient, long double v) = 0;

		// Reverse-вариант
		virtual long double forward_pass_Reverse(long double x) = 0;
		virtual void backward_pass_Reverse(long double gradient_output, std::vector<long double>& gradient_parameters, long double& gradient_input) = 0;
 
		// HyperDualReverse-вариант
		virtual dual::HyperDualCombination evaluate_HyperDualReverse(const dual::HyperDualCombination& x, const std::vector<size_t>& orders, int first_index) const = 0;
	};

	// Реализация полиномиальной функции степени m
	class Polynomial : public LearnableFunction {
	private:
		int m;
		std::vector<long double> a; // Коэффициенты
		int first_index; // Глобальный индекс первого коэффициента
		mutable long double last_input; // Последний вход для обратного прохода

	public:
		// Конструктор, задающий степень многочлена и инициализирующий коэффициенты как псевдослучайные числа
		Polynomial(int M) : m(M), first_index(-1), last_input(0.0L) {
			if (M < 0) throw std::invalid_argument("Degree must be non-negative");
			a.resize(M + 1);

			// Инициализация коэффициентов
			static std::mt19937 random_numbers(std::random_device{}());
			std::uniform_real_distribution<long double> dist(-0.5L, 0.5L);
			for (auto& coefficient : a) {
				coefficient = dist(random_numbers);
			}
		}

		// Возвращение количества параметров обучаемой полиномиальной функции
		int numParams() const override {
			return static_cast<int>(a.size());
		}

		// Установка глобального индекса первого коэффициента
		void setFirstParamIndex(int index) override {
			first_index = index;
		}

		// Чтение глобального индекса первого коэффициента
		int firstParamIndex() const override {
			return first_index;
		}

		// Вычисление многочлена от дуального числа x при прибавлении нильпотентного элемента к параметру с индексом j
		dual::Dual evaluate_Forward(const dual::Dual& x, int j) const override {
			// Определение локального индекса local_j проверкой попадания индекса j в диапазон индексов от first_index до first_index + numParams()
			bool param_is_active = (j >= first_index && j < first_index + numParams());
			int local_j = param_is_active ? j - first_index : -1;

			dual::Dual Sum(0.0, 0.0);
			dual::Dual x_power(1.0, 0.0);

			for (int i = 0; i <= m; ++i) {
				dual::Dual coefficient;
				if (i == local_j) {
					coefficient = dual::Dual(a[i], 1.0);
				}
				else {
					coefficient = dual::Dual(a[i], 0.0);
				}
				Sum += coefficient * x_power;
				x_power = x_power * x;
			}

			return Sum;
		}

		void update(const std::vector<long double>& gradient, long double v) override {
			if (gradient.size() != a.size()) {
				throw std::invalid_argument("Gradient size mismatch");
			}
			for (size_t i = 0; i < a.size(); ++i) {
				a[i] -= v * gradient[i];
			}
		}

		// Прямой проход по графу вычислений для Reverse Mode AD
		long double forward_pass_Reverse(long double x) override {
			last_input = x; 
			long double result = 0.0L;
			long double x_power = 1.0L;
			
			for (int i = 0; i <= m; ++i) {
				result += a[i] * x_power;
				x_power *= x;
			}

			return result;
		}
		
		// Обратный проход по графу вычислений для Reverse Mode AD
		void backward_pass_Reverse(long double gradient_output, std::vector<long double>& gradient_parameters, long double& gradient_input) override {
			long double x_power = 1.0L;
			gradient_input = 0.0L;

			// Производная по входному значению
			for (int i = 1; i <= m; ++i) {
				gradient_input += i * a[i] * x_power;
				x_power *= last_input;
			}

			gradient_input *= gradient_output;

			// Производные по параметрам
			x_power = 1.0L;

			for (int i = 0; i <= m; ++i) {
				gradient_parameters[i] += gradient_output * x_power;
				x_power *= last_input;
			}

		}

		// HyperDualReverse-вариант функции evaluate()
		dual::HyperDualCombination evaluate_HyperDualReverse(const dual::HyperDualCombination& x, const std::vector<size_t>& orders, int first_index) const override {
			dual::HyperDualCombination Sum(orders);
			dual::HyperDualCombination x_power(orders, 1.0L);

			int global_index;
			for (int i = 0; i <= m; ++i) {
				global_index = first_index + i;
				dual::HyperDualCombination hyperdual_coefficient = dual::Nilpotent_Add(a[i], orders, global_index);
				Sum += hyperdual_coefficient * x_power;
				x_power = x_power * x;
			}

			return Sum;
		}
	};
}