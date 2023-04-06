package main

func main() {

	layers := []LayerDef{
		{input_dim: 2, output_dim: 1, activation: Sigmoid},
	}
	x := Matrix{{1, 2, 3}, {2, 1, 3}}
	y := Matrix{{1, 2, 3}}
	net := NewNet(0.3, MSE, layers)
	net.Train(x, y, 1000)
}
