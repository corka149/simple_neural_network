use super::MathError;

pub fn multiply_vectors(first_vec: &Vec<Vec<f64>>,
                        second_vec: &Vec<Vec<f64>>)
                        -> Result<Vec<Vec<f64>>, MathError> {
    let mut product: Vec<Vec<f64>> = Vec::new();

    for first_row in first_vec {
        let mut result_row: Vec<f64> = Vec::new();

        for second_row in transpose_vec(&second_vec) {
            if first_row.len() != second_row.len() {
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

pub fn transpose_vec<T>(target: &Vec<Vec<T>>) -> Vec<Vec<T>>
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
                //TODO Danger Zone: Replace unwrap
                trans_vec.get_mut(index).unwrap().push(cell.clone());
            }
        }
        is_first_run = false;
    }
    trans_vec
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_vec() {
        let a_1: Vec<f64> = vec![1.0, 2.0];
        let a_2: Vec<f64> = vec![3.0, 4.0];
        let a_3: Vec<f64> = vec![5.0, 6.0];
        let a = vec![a_1, a_2, a_3];
        let new_a = transpose_vec(&a);

        assert_eq!(a[0][0], 1.0);
        assert_eq!(a[2][1], 6.0);
    }

    #[test]
    fn test_multiply_vecotrs() {
        let vec_a_1: Vec<f64> = vec![1.0, 2.0, 3.0];
        let vec_a_2: Vec<f64> = vec![4.0, 5.0, 6.0];
        let mut vec_a: Vec<Vec<f64>> = vec![vec_a_1, vec_a_2];

        let vec_b_1: Vec<f64> = vec![7.0, 8.0];
        let vec_b_2: Vec<f64> = vec![9.0, 10.0];
        let vec_b_3: Vec<f64> = vec![11.0, 12.0];
        let mut vec_b: Vec<Vec<f64>> = vec![vec_b_1, vec_b_2, vec_b_3];

        let result: Vec<Vec<f64>> = multiply_vectors(&vec_a, &vec_b).unwrap();

        assert_eq!(result[0][0], 58.0);
        assert_eq!(result[0][1], 64.0);
        assert_eq!(result[1][0], 139.0);
        assert_eq!(result[1][1], 154.0);
    }

    #[test]
    #[should_panic(expected = "incompatible vectors")]
    fn test_multiply_vectors_error() {
        let vec_a_1: Vec<f64> = vec![1.0, 2.0];
        let vec_a_2: Vec<f64> = vec![4.0, 5.0];
        let mut vec_a: Vec<Vec<f64>> = vec![vec_a_1, vec_a_2];

        let mut vec_b: Vec<Vec<f64>> = vec![vec![7.0, 8.0]];

        if let Err(e) = multiply_vectors(&vec_a, &vec_b) {
            panic!("{}", e);
        }
    }
}
