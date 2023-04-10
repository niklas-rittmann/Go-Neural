package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"neural-go/net"
)

func main() {

	fn := net.ActPersistMap["neural-go/net.relu"]
	log.Printf("%v", fn(-51))

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
	// Persist
	file, _ := json.MarshalIndent(model, "", " ")
	_ = ioutil.WriteFile("test.json", file, 0644)

	newNet := net.NewNetFromFile("test.json")

	newNet.Train(x, y, 10)
	fmt.Printf("Expected Preds %v\n", y)
	newNet.Predict(x)
}
