package net

type CostFunc interface {
	calcCost(y_hat, y Matrix) float64
	calcBackProp(y_hat, y Matrix) Matrix
}

type back_prop func(Matrix, Matrix) Matrix
type cost func(Matrix, Matrix) float64

type CostFunction struct {
	cost_func     cost
	backprop_func back_prop
}

func (c CostFunction) calcCost(y_hat, y Matrix) float64 {
	return c.cost_func(y_hat, y)
}
func (c CostFunction) calcBackProp(y_hat, y Matrix) Matrix {
	return c.backprop_func(y_hat, y)
}

func mse(y_hat, y Matrix) float64 {
	return sum(pow(diff(y_hat, y), 2.0))
}

func mse_back(y_hat, y Matrix) Matrix {
	return multiply_by(diff(y_hat, y), 2.0)
}

var MSE = CostFunction{mse, mse_back}
