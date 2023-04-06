package main

import "fmt"

func main() {

	layers := []LayerDef{
		{input_dim: 2, output_dim: 5, activation: ReLU},
		{input_dim: 5, output_dim: 1, activation: Sigmoid},
	}
	// Train Example
	//1, 2
	//2, 1
	//3,3

	//Predicts
	//0
	//0.5
	//1

	x := Matrix{{1, 2, 3}, {2, 1, 3}}
	y := Matrix{{0, 0.5, 1}}
	net := NewNet(0.3, MSE, layers)
	net.Train(x, y, 100000)
	fmt.Printf("Expected Preds %v\n", y[0])
	net.Predict(x)
}
