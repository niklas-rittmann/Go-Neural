package net

import "math"

type Activation func(float64) float64

type ActivationFunction struct {
	ActivationFunc Activation
	BackpropFunc   Activation
}

func (a ActivationFunction) calcActivation(matrix Matrix) Matrix {
	return wrapper(matrix, a.ActivationFunc)
}
func (a ActivationFunction) calcBackProp(matrix Matrix) Matrix {
	return wrapper(matrix, a.BackpropFunc)
}

func wrapper(matrix Matrix, fn Activation) Matrix {
	activated_matrix := Matrix{}
	for _, row := range matrix {
		activated_row := []float64{}
		for _, col := range row {
			val := fn(col)
			activated_row = append(activated_row, val)
		}
		activated_matrix = append(activated_matrix, activated_row)
	}
	return activated_matrix
}

func sigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}
func sigmoid_deriv(val float64) float64 {
	return sigmoid(val) * (1 - sigmoid(val))
}

func relu(val float64) float64 {
	if val < 0 {
		return 0
	}
	return val
}
func relu_back(val float64) float64 {
	if val < 0 {
		return 0
	}
	return val
}

var Sigmoid = ActivationFunction{sigmoid, sigmoid_deriv}
var ReLU = ActivationFunction{relu, relu_back}
