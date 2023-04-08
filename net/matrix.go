package net

import (
	"fmt"
	"math"
)

// Define a Matrix as 2D-Vektor
type Matrix [][]float64

// Return the shape of the current matrix
func (m *Matrix) shape() (int, int) {
	return len(*m), len((*m)[0])
}

// Stringify output of matrix
func (m *Matrix) String() string {
	y, x := (*m).shape()
	return fmt.Sprintf("%d x %d", y, x)
}

// Transpose  matrix
func (m Matrix) Transpose() Matrix {
	y, x := (m).shape()
	result := initEmptyMatrix(x, y)
	for i := 0; i < y; i++ {
		for j := 0; j < x; j++ {
			result[j][i] = m[i][j]
		}
	}
	return result
}

// Perform dot prodcuct between two matrices
func dot_product(mat_1, mat_2 Matrix) Matrix {
	n := len(mat_1)
	ma := len(mat_2)
	p := len(mat_2[0])

	result := initEmptyMatrix(n, p)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < ma; k++ {
				sum += (mat_1)[i][k] * mat_2[k][j]
			}
			result[i][j] = float64(sum)
		}
	}
	return result
}

// Calc the sum of each row (Transforms 2x3 into 2x1)
func row_sum(matrix Matrix) Matrix {
	result := Matrix{}
	for _, row := range matrix {
		sum := 0.0
		for _, val := range row {
			sum += val
		}
		result = append(result, []float64{sum})
	}
	return result
}

// Perform elementwise multiplikation with faktor
func multiply_by(matrix Matrix, factor float64) Matrix {
	for row_idx, row := range matrix {
		for col_idx := range row {
			matrix[row_idx][col_idx] *= factor
		}
	}
	return matrix
}

// Return Powered Matrix
func pow(matrix Matrix, factor float64) Matrix {
	for row_idx, row := range matrix {
		for col_idx := range row {
			matrix[row_idx][col_idx] = math.Pow(matrix[row_idx][col_idx], factor)
		}
	}
	return matrix
}

// Perform elementwise multiplikation
func multiply(mat_1, mat_2 Matrix) Matrix {
	fn := func(x, y float64) float64 { return x * y }
	return elementwise_matrix_op(mat_1, mat_2, fn)
}

// Perform elementwise multiplikation
func divide(mat_1, mat_2 Matrix) Matrix {
	fn := func(x, y float64) float64 { return x / y }
	return elementwise_matrix_op(mat_1, mat_2, fn)
}

// Add two matrices together
func add(mat_1, mat_2 Matrix) Matrix {
	fn := func(x, y float64) float64 { return x + y }
	return elementwise_matrix_op(mat_1, mat_2, fn)
}

// Subtract two matrices together
func diff(mat_1, mat_2 Matrix) Matrix {
	fn := func(x, y float64) float64 { return x - y }
	return elementwise_matrix_op(mat_1, mat_2, fn)
}

func elementwise_matrix_op(mat_1, mat_2 Matrix, fn func(x, y float64) float64) Matrix {
	y, x := mat_1.shape()
	result := initEmptyMatrix(y, x)
	for row := 0; row < y; row++ {
		for col := 0; col < x; col++ {
			mat_1_val := mat_1[row][col]
			mat_2_val := mat_2[row][col]
			result[row][col] = fn(mat_1_val, mat_2_val)
		}
	}
	return result
}

// Add a vector to matrix
func add_vector(mat_1, mat_2 Matrix) Matrix {
	result := Matrix{}
	for row_idx, row := range mat_1 {
		row_vals := []float64{}
		for _, mat1_val := range row {
			sum := mat1_val + mat_2[row_idx][0]
			row_vals = append(row_vals, sum)
		}
		result = append(result, row_vals)
	}
	return result
}

// Calc the sum of a matrix (used for cost)
func sum(matrix Matrix) float64 {
	sum := 0.0
	for _, row := range matrix {
		for _, val := range row {
			sum += val
		}
	}
	return sum
}

func initEmptyMatrix(n, m int) Matrix {
	result := Matrix{}
	for i := 0; i < n; i++ {
		row := make([]float64, m)
		result = append(result, row)
	}
	return result
}
