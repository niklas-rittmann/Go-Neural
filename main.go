package main

import (
	"fmt"
	"neural-go/net"
)

func main() {

	layers := []net.LayerDef{
		{Input_dim: 2, Output_dim: 25, Activation: net.ReLU},
		{Input_dim: 25, Output_dim: 2, Activation: net.Sigmoid},
	}
	// Train Example
	//1,		2
	//2,		1
	//3,		3

	//Predicts
	//0,		1
	//0.5,	0.5
	//1,		3

	x := net.Matrix{
		{1, 2},
		{2, 1},
		{3, 3},
	}.Transpose()

	y := net.Matrix{
		{0, 1},
		{0.5, 0.5},
		{1, 1},
	}.Transpose()
	model := net.NewNet(0.5, net.MSE, layers)
	model.Train(x, y, 10)
	model.ToFile("2layer.json")
	fmt.Printf("Expected Preds %v\n", y)
	model.Predict(x)
}
