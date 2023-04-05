package main

import "math"

type ActFunction interface {
	calcActivation(matrix Matrix) Matrix
	calcBackProp(matrix Matrix) Matrix
}

type activation func(float64) float64

type ActivationFunction struct {
	activation_func activation
	backprop_func   activation
}

func (a ActivationFunction) calcActivation(matrix Matrix) Matrix {
	return activation_wrapper(matrix, a.activation_func)
}
func (a ActivationFunction) calcBackProp(matrix Matrix) Matrix {
	return [][]float64{}
}

func activation_wrapper(matrix Matrix, fn activation) Matrix {
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
func sigmoid_back(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
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

var Sigmoid = ActivationFunction{sigmoid, sigmoid_back}
var ReLU = ActivationFunction{relu, relu_back}