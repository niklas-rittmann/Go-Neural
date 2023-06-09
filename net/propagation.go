package net

import "log"

type BackPropRes struct {
	dw Matrix
	db Matrix
}

// Perform Training with the given net
func (n *Net) Train(x, y Matrix, epochs int) {
	for i := 0; i < epochs; i++ {
		a_curr := n.forward(x)
		n.backward(a_curr, y)
		log.Printf("Cost in Epoch %d: %f", i, n.Cost.CalcCost(a_curr, y))
	}
}

// Perform Predict with the given net
func (n *Net) Predict(x Matrix) Matrix {
	a_curr := n.forward(x)
	log.Printf("Predict %v", a_curr.Transpose())
	return a_curr
}

// Perform forward propagation
func (n *Net) forward(x Matrix) Matrix {
	var z_curr Matrix
	var a_curr Matrix = x

	for layer_idx, layer := range n.Layers {
		a_prev := a_curr
		a_curr, z_curr = single_layer_forward(a_prev, layer)
		n.cache[layer_idx] = CacheVal{a_prev, z_curr}
	}
	return a_curr
}

// Perform forward propagation for a single layer
func single_layer_forward(a Matrix, layer Layer) (Matrix, Matrix) {
	weightet_in := dot_product(layer.Weights, a)
	z_curr := add_vector(weightet_in, layer.Bias)
	return layer.Activation.calcActivation(z_curr), z_curr
}

// Perform backward propagation for current layer
func (n *Net) backward(y_hat, y Matrix) {
	var da_prev = n.Cost.calcBackProp(y_hat, y)
	var da_curr Matrix
	var derivs BackPropRes

	for layer_idx := len((*n).Layers) - 1; layer_idx >= 0; layer_idx-- {
		layer := n.Layers[layer_idx]
		cache := n.cache[layer_idx]
		da_curr = da_prev
		da_prev, derivs = single_layer_backward(da_curr, layer, cache)

		n.Layers[layer_idx].Weights = diff(layer.Weights, multiply_by(derivs.dw, n.Lr))
		n.Layers[layer_idx].Bias = diff(layer.Bias, multiply_by(derivs.db, n.Lr))
	}
}

// Calculate derivitives for a single layer
func single_layer_backward(da_curr Matrix, layer Layer, cache CacheVal) (Matrix, BackPropRes) {
	_, y := da_curr.shape()
	dz_curr := multiply(da_curr, layer.Activation.calcBackProp(cache.z))

	dw_curr := multiply_by(dot_product(dz_curr, cache.a.Transpose()), 1.0/float64(y))

	db_curr := multiply_by(row_sum(dz_curr), 1.0/float64(y))

	da_prev := dot_product(layer.Weights.Transpose(), dz_curr)
	return da_prev, BackPropRes{dw_curr, db_curr}
}
