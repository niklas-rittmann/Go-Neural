package net

import "math/rand"

// This type for describing the net structure
type LayerDef struct {
	Input_dim  int
	Output_dim int
	Activation ActFunction
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
		weights := InitMatrix(layer.Output_dim, layer.Input_dim)
		bias := InitMatrix(layer.Output_dim, 1)
		layers = append(layers, Layer{weights, bias, layer.Activation})
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
