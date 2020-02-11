# Fitting with MLP using TensorFlow
Goal of this repository is to share programs that fit some kinds of curves by high configurable multilayer perceptron (MLP) neural network written in Python 3 using TensorFlow 2 with Keras.

## One variable function fitting
The project [One variable function fitting](./one-variable-function-fitting) implements the fitting of a continuous and limited real-valued function defined in a closed interval of the reals.
Please visit (here)[https://computationalmindset.com/en/posts/neural-networks/one-variable-function-fitting-with-tensorflow.html] for concepts about this project.

## Parametric curve on plane fitting
The project [Parametric curve on plane fitting](./parametric-curve-on-plane-fitting) implements the fitting of a continuous and limited real-valued parametric curve on plane where parameter belongs to a closed interval of the reals. It implements two alternative techniques: the official one implements one MLP that fits a vector function f(t) = [x(t), y(t)] instead the 'twin' variant implements a pair of twins of MLPs that fit separately the one variable functions x(t) and y(t).
Please visit [here](https://computationalmindset.com/en/posts/neural-networks/parametric-curve-on-plane-fitting-with-tensorflow.html) for concepts about this project.

## Parametric curve on space fitting
The project [Parametric curve on space fitting](./parametric-curve-on-space-fitting) implements the fitting of a continuous and limited real-valued parametric curve on space where parameter belongs to a closed interval of the reals.
