package net

type BackProp func(Matrix, Matrix) Matrix
type Cost func(Matrix, Matrix) float64

type CostFunction struct {
	CostFunc     Cost
	BackpropFunc BackProp
}

func (c CostFunction) CalcCost(y_hat, y Matrix) float64 {
	return c.CostFunc(y_hat, y)
}
func (c CostFunction) calcBackProp(y_hat, y Matrix) Matrix {
	return c.BackpropFunc(y_hat, y)
}

func mse(y_hat, y Matrix) float64 {
	return sum(pow(diff(y_hat, y), 2.0))
}

func mse_back(y_hat, y Matrix) Matrix {
	return multiply_by(diff(y_hat, y), 2.0)
}

var MSE = CostFunction{mse, mse_back}
