package main

import (
	"reflect"
	"testing"
)

func TestMatrixMultiplication(t *testing.T) {
	arr1 := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	arr2 := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	got := multiply(arr1, arr2)
	want := Matrix{{6.0, 12.0, 18.0}, {6.0, 12.0, 18.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v but wanted %#v", got, want)
	}
}
func TestAddMatrices(t *testing.T) {
	t.Run("Quadratic Matrix", func(t *testing.T) {
		arr1 := Matrix{{1.0, 2.0}, {1.0, 2.0}}
		arr2 := Matrix{{1.0, 2.0}, {1.0, 2.0}}
		got := add(arr1, arr2)
		want := Matrix{{2.0, 4.0}, {2.0, 4.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %#v but wanted %#v", got, want)
		}
	})
	t.Run("Column Vector", func(t *testing.T) {
		arr1 := Matrix{{1.0}, {1.0}}
		arr2 := Matrix{{2.0}, {2.0}}
		got := add(arr1, arr2)
		want := Matrix{{3.0}, {3.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %#v but wanted %#v", got, want)
		}
	})

}

func TestInitEmptyMatrix(t *testing.T) {
	want := Matrix{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}
	got := initEmptyMatrix(2, 3)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v but wanted %v", got, want)
	}
}
