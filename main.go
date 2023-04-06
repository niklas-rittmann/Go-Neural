package net

import (
	"fmt"
	"neural-go/net"
)

func main() {

	layers := []net.LayerDef{
		{Input_dim: 2, Output_dim: 5, Activation: net.ReLU},
		{Input_dim: 5, Output_dim: 1, Activation: net.Sigmoid},
	}
	// Train Example
	//1, 2
	//2, 1
	//3,3

	//Predicts
	//0
	//0.5
	//1

	x := net.Matrix{{1, 2, 3}, {2, 1, 3}}
	y := net.Matrix{{0, 0.5, 1}}
	net := net.NewNet(0.3, net.MSE, layers)
	net.Train(x, y, 100000)
	fmt.Printf("Expected Preds %v\n", y[0])
	net.Predict(x)
}
