package main

import "math"

type CostFunc interface {
	calcCost(matrix Matrix) float64
	calcBackProp(matrix Matrix) Matrix
}

type cost func(float64, float64) float64

type CostFunction struct {
	cost_func     cost
	backprop_func cost
}

func (c CostFunction) calcCost(matrix Matrix) float64 {
	return 0.0
}
func (c CostFunction) calcBackProp(matrix Matrix) Matrix {
	return [][]float64{}
}

func cost_wrapper(y_hat, y Matrix, fn cost) float64 {
	counter := 0
	cost := 0.0
	for row_idx, row := range y {
		for col_idx, y_val := range row {
			counter += 1
			y_hat_val := y_hat[row_idx][col_idx]
			cost += fn(y_hat_val, y_val)
		}
	}
	return cost / float64(counter)
}

func mse(y_hat, y float64) float64 {
	return math.Pow(y-y_hat, 2)
}

func mse_back(y_hat, y float64) float64 {
	return 2 * (y - y_hat)
}

var MSE = CostFunction{mse, mse_back}
