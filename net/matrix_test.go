package net

import (
	"reflect"
	"testing"
)

func TestMatrixShape(t *testing.T) {
	arr := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	got_x, got_y := arr.shape()
	want_x, want_y := 2, 3
	if got_x != want_x {
		t.Errorf("got %d but wanted %d", got_x, want_x)
	}
	if got_y != want_y {
		t.Errorf("got %d but wanted %d", got_y, want_y)
	}
}

func TestMatrixString(t *testing.T) {
	arr := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	got := arr.String()
	want := "2 x 3"
	if got != want {
		t.Errorf("got %s but wanted %s", got, want)
	}
}
func TestMatrixTranspose(t *testing.T) {
	arr := Matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}
	got := arr.Transpose()
	want := Matrix{{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v but wanted %v", got, want)
	}
}

func TestMatrixDotProdcut(t *testing.T) {
	arr1 := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	arr2 := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	got := dot_product(arr1, arr2)
	want := Matrix{{6.0, 12.0, 18.0}, {6.0, 12.0, 18.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v but wanted %#v", got, want)
	}
}

func TestRowSum(t *testing.T) {
	arr := Matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}
	got := row_sum(arr)
	want := Matrix{{6.0}, {15.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v but wanted %#v", got, want)
	}
}

func TestMatrixMultiplicationBy(t *testing.T) {
	arr := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	factor := 2.0
	got := multiply_by(arr, factor)
	want := Matrix{{2.0, 4.0, 6.0}, {2.0, 4.0, 6.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v but wanted %#v", got, want)
	}
}
func TestMatrixMultiplication(t *testing.T) {
	mat_1 := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	mat_2 := Matrix{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}}
	got := multiply(mat_1, mat_2)
	want := mat_1
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v but wanted %#v", got, want)
	}
}
func TestMatrixPow(t *testing.T) {
	arr := Matrix{{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
	factor := 2.0
	got := pow(arr, factor)
	want := Matrix{{1.0, 4.0, 9.0}, {1.0, 4.0, 9.0}}
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
func TestAddVector(t *testing.T) {
	t.Run("Add vector to matrix", func(t *testing.T) {
		arr1 := Matrix{{1.0, 2.0}, {1.0, 2.0}}
		arr2 := Matrix{{1.0}, {1.0}}
		got := add_vector(arr1, arr2)
		want := Matrix{{2.0, 3.0}, {2.0, 3.0}}
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
