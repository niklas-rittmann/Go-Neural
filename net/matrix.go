package main

// Define a Matrix as 2D-Vektor
type Matrix [][]float64

// Perform dot prodcuct between two matrices
func multiply(mat_1, mat_2 Matrix) Matrix {
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

// Add two matrices together
func add(mat_1, mat_2 Matrix) Matrix {
	result := Matrix{}
	for row_idx, row := range mat_1 {
		row_vals := []float64{}
		for col_idx, mat1_val := range row {
			sum := mat1_val + mat_2[row_idx][col_idx]
			row_vals = append(row_vals, sum)
		}
		result = append(result, row_vals)
	}
	return result
}

func initEmptyMatrix(n, m int) Matrix {
	result := Matrix{}
	row := make([]float64, m)
	for i := 0; i < n; i++ {
		result = append(result, row)
	}
	return result
}
