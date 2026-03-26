DESCRIPTION
KAN is a C++ library for creating and training Kolmogorov‑Arnold Networks (KAN) using automatic differentiation. Three training modes are implemented:
Forward Mode AD (direct pass with dual numbers)
Reverse Mode AD (classical reverse pass)
HyperDualReverse Mode AD (author's method based on hyper‑dual combinations)
The library demonstrates the application of automatic differentiation in machine learning. Currently, polynomials are used as learnable functions; support for other elementary functions is planned.

FEATURES
Template classes LearnableFunction, Polynomial (easily extensible).
KAN_Layer layer.
Full KAN network.
Training functions for each mode: Step_Forward, Step_Reverse, Step_HyperDualReverse.
Loss evaluation functions.
Arbitrary number of layers and neurons.

INSTALLATION AND BUILD
The library is header‑only. To use, copy the KAN folder into your project and include the required headers:

#include "KAN/KAN_network.h"

All classes are in namespace kan.

DEPENDENCIES
RealUtils
Rational (required by Dual)
Dual (core AD library)

EXAMPLE (SINE APPROXIMATION)

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

LICENSE
MIT License. See the LICENSE file.

AUTHOR
Mikhail D. Sychev
Email: murzilkabest@icloud.com
Telegram: @Murz1k22
