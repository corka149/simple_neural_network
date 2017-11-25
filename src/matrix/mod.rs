pub mod math;
pub mod error;
use super::util::*;

pub struct Matrix {
    rows: usize,
    columns: usize,
    data_container: Vec<Vec<f64>>,
}

impl Matrix {

    pub fn zero(rows: usize, columns: usize) -> Matrix {
        let mut data_container: Vec<Vec<f64>> = Vec::new();

        for _row_count in 0..rows {
            data_container.push(math::create_zeroed_vector(columns));
        }

        Matrix{
            rows,
            columns,
            data_container,
        }
    }

    pub fn create_weighting_matrix(x: u64, y: u64) -> Matrix {
        let mut weighting_vec = Vec::new();

        for _number in 0..y {
            weighting_vec.push(create_weighting_row(x));
        }

        weighting_vec
    }

    pub fn multiply(&self, right: &Matrix) -> Result<Matrix,error::MathError> {
        let result = math::multiply_matrices(&self.data_container, &right.data_container)?;

        Ok(Matrix {
            rows: self.rows,
            columns: right.columns,
            data_container: result,
        })
    }

    pub fn add(&self, right: &Matrix) -> Result<Matrix,error::MathError> {
        let result = math::sum_matrices(&self.data_container, &right.data_container)?;

        Ok(Matrix {
            rows: right.rows,
            columns: right.columns,
            data_container: result,
        })
    }
}



#[cfg(test)]
mod matrix_tests {
    use super::*;

    #[test]
    fn test_zero(){
        let m = Matrix::zero(2, 2);
        assert_eq!(m.rows, 2);
        assert_eq!(m.columns, 2);
        assert_eq!(m.data_container[0][0], 0.0);
        assert_eq!(m.data_container[1][1], 0.0);
    }

    #[test]
    fn test_create_weighting_vec() {
        let vec = Matrix::create_weighting_matrix(3, 4);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0].len(), 3);
    }

    #[test]
    fn test_multiply(){
        let mut m1 = Matrix::zero(2,3);
        m1.data_container[0][0] = 2.0;
        m1.data_container[0][1] = 3.0;
        m1.data_container[0][2] = 1.0;
        m1.data_container[1][0] = 2.0;
        m1.data_container[1][1] = 1.0;
        m1.data_container[1][2] = 1.0;

        let mut m2 = Matrix::zero(3, 2);
        m2.data_container[0][0] = 3.0;
        m2.data_container[0][1] = 3.0;
        m2.data_container[1][0] = 2.0;
        m2.data_container[1][1] = 1.0;
        m2.data_container[2][0] = 2.0;
        m2.data_container[2][1] = 1.0;

        let m3 = m1.multiply(&m2).unwrap();
        assert_eq!(m3.data_container[0][0], 14.0);
        assert_eq!(m3.data_container[0][1], 10.0);
        assert_eq!(m3.data_container[1][0], 10.0);
        assert_eq!(m3.data_container[1][1], 8.0);

        assert_eq!(m3.rows, 2);
        assert_eq!(m3.columns, 2);
    }

    #[test]
    fn test_multiply_err(){
        let mut m1 = Matrix::zero(2,3);
        m1.data_container[0][0] = 2.0;
        m1.data_container[0][1] = 3.0;
        m1.data_container[0][2] = 1.0;
        m1.data_container[1][0] = 2.0;
        m1.data_container[1][1] = 1.0;
        m1.data_container[1][2] = 1.0;

        let mut m2 = Matrix::zero(2, 2);
        m2.data_container[0][0] = 3.0;
        m2.data_container[0][1] = 3.0;
        m2.data_container[1][0] = 2.0;
        m2.data_container[1][1] = 1.0;

        let m3 = m1.multiply(&m2);

        assert!(m3.is_err());
    }

    #[test]
    fn test_add(){
        let mut m1 = Matrix::zero(2,2);
        m1.data_container[0][0] = 2.0;
        m1.data_container[0][1] = 3.0;
        m1.data_container[1][0] = 2.0;
        m1.data_container[1][1] = 1.0;

        let mut m2 = Matrix::zero(2, 2);
        m2.data_container[0][0] = 3.0;
        m2.data_container[0][1] = 3.0;
        m2.data_container[1][0] = 2.0;
        m2.data_container[1][1] = 1.0;

        let m3 = m1.add(&m2).unwrap();

        assert_eq!(m3.data_container[0][0], 5.0);
        assert_eq!(m3.data_container[0][1], 6.0);
        assert_eq!(m3.data_container[1][0], 4.0);
        assert_eq!(m3.data_container[1][1], 2.0);

        assert_eq!(m3.rows, 2);
        assert_eq!(m3.columns, 2);
    }

    #[test]
    fn test_add_err(){
        let mut m1 = Matrix::zero(2,2);
        m1.data_container[0][0] = 2.0;
        m1.data_container[0][1] = 3.0;
        m1.data_container[1][0] = 2.0;
        m1.data_container[1][1] = 1.0;

        let mut m2 = Matrix::zero(2, 1);
        m2.data_container[0][0] = 3.0;
        m2.data_container[1][0] = 3.0;

        let m3 = m1.add(&m2);

        assert!(m3.is_err());
    }
}
