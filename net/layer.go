package main

import "math/rand"

// This type for describing the net structure
type LayerDef struct {
	input_dim  int
	output_dim int
	activation ActFunction
}

// Type for each layer. Each layer consits if a weights and bias matrix as
// well as an activation function
type Layer struct {
	weights    Matrix
	bias       Matrix
	activation ActFunction
}

// Function to create layers from definition
func InitLayers(layer_defs []LayerDef) []Layer {
	layers := []Layer{}
	for _, layer := range layer_defs {
		weights := InitMatrix(layer.input_dim, layer.output_dim)
		bias := InitMatrix(layer.input_dim, 1)
		layers = append(layers, Layer{weights, bias, layer.activation})
	}
	return layers
}

// Randomly initalize Matrix
// Multiplying by 0.1 to reduce outbreaks
func InitMatrix(in, out int) Matrix {
	matrix := [][]float64{}
	for y := 0; y < in; y++ {
		cols := []float64{}
		for x := 0; x < out; x++ {
			cols = append(cols, rand.NormFloat64()*0.1)
		}
		matrix = append(matrix, cols)
	}
	return matrix
}
