use super::error::MathError;

pub fn multiply_matrices(left: &[Vec<f64>],
                         right: &[Vec<f64>])
                         -> Result<Vec<Vec<f64>>, MathError> {
    let mut product: Vec<Vec<f64>> = Vec::new();

    for first_row in left {
        let mut result_row: Vec<f64> = Vec::new();

        for second_row in transpose_matrix(right) {
            if first_row.len() != second_row.len() {
                eprintln!("left {} | right {}", first_row.len(), second_row.len());
                eprintln!("right matrix: {:?}", right);
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

pub fn subtract_vectors(left: &[f64], right: &[f64]) -> Vec<f64> {
    left
        .iter()
        .zip(right)
        .map(|(a, b)| a - b)
        .collect()
}

pub fn sum_matrices(left: &[Vec<f64>],
                    right: &[Vec<f64>])
                    -> Result<Vec<Vec<f64>>, MathError> {
    let mut x = 0;
    let mut result: Vec<Vec<f64>> = Vec::new();

    if left.len() != right.len() {
        return Err(MathError);
    }

    while x < left.len() {
        let row_first: &Vec<f64> = match left.get(x) {
            Some(r) => r,
            None => return Err(MathError),
        };
        let row_second: &Vec<f64> = match right.get(x) {
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
pub fn from_vector_to_matrix(target_vec: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
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

pub fn create_zeroed_vector(columns: usize) -> Vec<f64> {
    let mut index = 0;
    let mut zeroed_vec: Vec<f64> = Vec::new();

    while index < columns {
        zeroed_vec.push(0.0);

        index += 1;
    }

    zeroed_vec
}

#[cfg(test)]
mod math_tests {
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
    fn test_from_vector_to_matrix() {
        let test_vec = vec![vec![1.0], vec![2.0]];
        assert_eq!(test_vec[0][0], 1.0);
        assert_eq!(test_vec[1][0], 2.0);
        let test_vec = from_vector_to_matrix(&test_vec).unwrap();
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

    use super::*;

    #[test]
    fn test_create_zeroed_vector() {
        let z_vec = create_zeroed_vector(3);
        assert_eq!(z_vec[0], 0.0);
        assert_eq!(z_vec[1], 0.0);
        assert_eq!(z_vec[2], 0.0);
    }
}
