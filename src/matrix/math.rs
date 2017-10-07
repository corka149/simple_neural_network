use super::MathError;
use super::*;

pub fn multiply_matrices(first_vec: &[Vec<f64>],
                         second_vec: &[Vec<f64>])
                         -> Result<Vec<Vec<f64>>, MathError> {
    let mut product: Vec<Vec<f64>> = Vec::new();

    for first_row in first_vec {
        let mut result_row: Vec<f64> = Vec::new();

        for second_row in transpose_matrix(second_vec) {
            if first_row.len() != second_row.len() {
                eprintln!("left {} | right {}", first_row.len(), second_row.len());
                eprintln!("right matrix: {:?}", second_vec);
                return Err(MathError);
            }

            let cell = first_row
                .iter()
                .zip(second_row)
                .map(|(a, b)| a * b)
                .sum();
            result_row.push(cell);
        }
        product.push(result_row);
    }

    Ok(product)
}

pub fn subtract_vectors(first_vec: &[f64], second_vec: &[f64]) -> Vec<f64> {
    first_vec
        .iter()
        .zip(second_vec)
        .map(|(a, b)| a - b)
        .collect()
}

pub fn sum_matrices(first: &[Vec<f64>],
                    second: &[Vec<f64>])
                    -> Result<Vec<Vec<f64>>, MathError> {
    let mut x = 0;
    let mut result: Vec<Vec<f64>> = Vec::new();

    if first.len() != second.len() {
        return Err(MathError);
    }

    while x < first.len() {
        let row_first: &Vec<f64> = match first.get(x) {
            Some(r) => r,
            None => return Err(MathError),
        };
        let row_second: &Vec<f64> = match second.get(x) {
            Some(r) => r,
            None => return Err(MathError),
        };

        if row_first.len() != row_second.len() {
            return Err(MathError);
        }

        let row = row_first
            .iter()
            .zip(row_second)
            .map(|(a, b)| a + b)
            .collect();

        result.push(row);
        x += 1;
    }

    Ok(result)
}

pub fn transpose_matrix<T>(target: &[Vec<T>]) -> Vec<Vec<T>>
    where T: Clone
{
    let mut trans_vec: Vec<Vec<T>> = Vec::new();
    let mut is_first_run = true;

    for line in target {
        for (index, cell) in line.iter().enumerate() {
            if is_first_run {
                let mut column = Vec::new();
                column.push(cell.clone());
                trans_vec.push(column);
            } else {
                let val = match trans_vec.get_mut(index) {
                    Some(v) => v,
                    None => panic!("Nothing here"),
                };
                val.push(cell.clone());
            }
        }
        is_first_run = false;
    }
    trans_vec
}

/// Example:    1       1   0   0
///             2   =>  0   2   0
///             3       0   0   3
pub fn make_vector_quadratic(target_vec: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let mut quadratic_vec: Vec<Vec<f64>> = vec![];

    for (index, value) in target_vec.iter().enumerate() {
        let row = create_zeroed_vector(target_vec.len());
        quadratic_vec.push(row);

        let value = match value.get(0) {
            Some(v) => v,
            None => panic!("Vector is empty - invalid state!"),
        };

        let row = match quadratic_vec.get_mut(index) {
            Some(r) => r,
            None => return None,
        };

        if let Some(v) = row.get_mut(index) {
            *v = *value;
        };
    }

    Some(quadratic_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_matrix() {
        let a_1: Vec<f64> = vec![1.0, 2.0];
        let a_2: Vec<f64> = vec![3.0, 4.0];
        let a_3: Vec<f64> = vec![5.0, 6.0];
        let a = vec![a_1, a_2, a_3];
        let new_a = transpose_matrix(&a);

        assert_eq!(a[0][0], 1.0);
        assert_eq!(a[2][1], 6.0);
    }

    #[test]
    fn test_multiply_matrices() {
        let vec_a_1: Vec<f64> = vec![1.0, 2.0, 3.0];
        let vec_a_2: Vec<f64> = vec![4.0, 5.0, 6.0];
        let mut vec_a: Vec<Vec<f64>> = vec![vec_a_1, vec_a_2];

        let vec_b_1: Vec<f64> = vec![7.0, 8.0];
        let vec_b_2: Vec<f64> = vec![9.0, 10.0];
        let vec_b_3: Vec<f64> = vec![11.0, 12.0];
        let mut vec_b: Vec<Vec<f64>> = vec![vec_b_1, vec_b_2, vec_b_3];

        let result: Vec<Vec<f64>> = multiply_matrices(&vec_a, &vec_b).unwrap();

        assert_eq!(result[0][0], 58.0);
        assert_eq!(result[0][1], 64.0);
        assert_eq!(result[1][0], 139.0);
        assert_eq!(result[1][1], 154.0);
    }

    #[test]
    #[should_panic(expected = "incompatible vectors")]
    fn test_multiply_matrices_error() {
        let vec_a_1: Vec<f64> = vec![1.0, 2.0];
        let vec_a_2: Vec<f64> = vec![4.0, 5.0];
        let mut vec_a: Vec<Vec<f64>> = vec![vec_a_1, vec_a_2];

        let mut vec_b: Vec<Vec<f64>> = vec![vec![7.0, 8.0]];

        if let Err(e) = multiply_matrices(&vec_a, &vec_b) {
            panic!("{}", e);
        }
    }

    #[test]
    fn test_subtract_vectors() {
        let vec_a = vec![2.0, 3.0];
        let vec_b = vec![1.0, 1.5];
        let result = subtract_vectors(&vec_a, &vec_b);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 1.5);
    }

    #[test]
    fn test_make_vector_quadratic() {
        let test_vec = vec![vec![1.0], vec![2.0]];
        assert_eq!(test_vec[0][0], 1.0);
        assert_eq!(test_vec[1][0], 2.0);
        let test_vec = make_vector_quadratic(&test_vec).unwrap();
        assert_eq!(test_vec[0][0], 1.0);
        assert_eq!(test_vec[1][1], 2.0);
    }

    #[test]
    fn test_sum_matrices() {
        let first = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let second = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = sum_matrices(&first, &second).unwrap();
        assert_eq!(result[0][0], 6.0);
        assert_eq!(result[0][1], 8.0);
        assert_eq!(result[1][0], 10.0);
        assert_eq!(result[1][1], 12.0);
    }
}
