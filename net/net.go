package main

import "log"

// Struct for creating a Neural-Network
type Net struct {
	lr     float64
	layers []Layer
	cache  map[int]CacheVal
}

type CacheVal struct {
	a Matrix
	z Matrix
}

// Perform Training with the given net
func (n *Net) Train(x, y Matrix, epochs int) {
	for i := 0; i < epochs; i++ {
		//a_curr := n.forward(x)

	}

}

// Perform forward propagation
func (n *Net) forward(x Matrix) Matrix {
	var z_curr Matrix
	var a_curr Matrix = x

	for layer_idx, layer := range n.layers {
		a_prev := a_curr
		a_curr, z_curr = single_layer_forward(a_prev, layer)
		n.cache[layer_idx] = CacheVal{a_prev, z_curr}
	}
	return a_curr
}

// Perform forward propagation for a single layer
func single_layer_forward(a Matrix, layer Layer) (Matrix, Matrix) {
	weightet_in := multiply(layer.weights, a)
	log.Printf("Got matrix %v", weightet_in)
	z_curr := add(weightet_in, layer.bias)
	return layer.activation.calcActivation(z_curr), z_curr
}

func (n *Net) Backward() {

}

// Create a new Net based on the provided input
func NewNet(lr float64, layer_defs []LayerDef) *Net {
	layers := InitLayers(layer_defs)
	return &Net{lr: lr, layers: layers}
}
