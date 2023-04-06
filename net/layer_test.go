package net

import "testing"

func TestInitLayers(t *testing.T) {
	layers := []LayerDef{{Input_dim: 4, Output_dim: 3, Activation: Act{}}}
	t.Run("Num of arrays", func(t *testing.T) {
		got := len(InitLayers(layers))
		want := 1
		if got != want {
			t.Errorf("got %d but wanted %d", got, want)
		}
	})
	t.Run("Activation stil lthe same", func(t *testing.T) {
		got := InitLayers(layers)[0].activation
		want := layers[0].Activation
		if got != want {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
}

func TestInitMatrix(t *testing.T) {
	t.Run("Num of rows", func(t *testing.T) {
		got := len(InitMatrix(1, 5))
		want := 1
		if got != want {
			t.Errorf("got %d but wanted %d", got, want)
		}
	})
	t.Run("Num of cols", func(t *testing.T) {
		got := len(InitMatrix(1, 5)[0])
		want := 5
		if got != want {
			t.Errorf("got %d but wanted %d", got, want)
		}
	})
	t.Run("Vals in range", func(t *testing.T) {
		got := InitMatrix(1, 5)[0][0]
		if got < -1 || got > 1 {
			t.Errorf("Value %f is outside range -1 and 1", got)
		}
	})
}
