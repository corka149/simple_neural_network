pub mod math;
pub mod error;

pub struct Matrix {
    rows: usize,
    columns: usize,
    data_container: Vec<Vec<f64>>,
}

impl Matrix {

    fn zero(rows: usize, columns: usize) -> Matrix {
        let mut data_container: Vec<Vec<f64>> = Vec::new();

        for _times in 0..data_container {
            data_container.push(create_zeroed_vector(columns));
        }

        Matrix{
            rows,
            columns,
            data_container,
        }
    }

    fn multiply(&self, right: &Matrix) -> Result<Matrix,error::MathError> {
        let result = math::multiply_matrices(self.data_container, right.data_container)?;

        Matrix {
            rows: right.rows,
            columns: right.columns,
            data_container: result,
        }
    }

    fn add(&self, right: &Matrix) -> Result<Matrix,error::MathError> {
        let result = math::sum_matrices(&self.data_container, &right.data_container)?;

        Matrix {
            rows: right.rows,
            columns: right.columns,
            data_container: result,
        }
    }
}

pub fn create_zeroed_vector(length: usize) -> Vec<f64> {
    let mut index = 0;
    let mut zeroed_vec: Vec<f64> = Vec::new();

    while index < length {
        zeroed_vec.push(0.0);

        index += 1;
    }

    zeroed_vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_zeroed_vector() {
        let z_vec = create_zeroed_vector(3);
        assert_eq!(z_vec[0], 0.0);
        assert_eq!(z_vec[1], 0.0);
        assert_eq!(z_vec[2], 0.0);
    }
}
