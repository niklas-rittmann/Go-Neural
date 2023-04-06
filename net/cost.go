package net

type CostFunc interface {
	calcCost(y_hat, y Matrix) Matrix
	calcBackProp(y_hat, y Matrix) Matrix
}

type cost func(Matrix, Matrix) Matrix

type CostFunction struct {
	cost_func     cost
	backprop_func cost
}

func (c CostFunction) calcCost(y_hat, y Matrix) Matrix {
	return c.cost_func(y_hat, y)
}
func (c CostFunction) calcBackProp(y_hat, y Matrix) Matrix {
	return c.backprop_func(y_hat, y)
}

func mse(y_hat, y Matrix) Matrix {
	return pow(diff(y_hat, y), 2.0)
}

func mse_back(y_hat, y Matrix) Matrix {
	return multiply_by(diff(y_hat, y), 2.0)
}

var MSE = CostFunction{mse, mse_back}
