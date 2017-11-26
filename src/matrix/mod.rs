pub mod math;
pub mod error;
use super::util::*;

pub struct Matrix {
    rows: usize,
    columns: usize,
    data_container: Vec<Vec<f64>>,
}

impl Matrix {

    pub fn zero(columns: usize, rows: usize) -> Matrix {
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

    pub fn create_weighting_matrix(columns: usize, rows: usize) -> Matrix {
        let mut weighting_vec: Vec<Vec<f64>> = Vec::new();

        for _number in 0..rows {
            weighting_vec.push(create_weighting_row(columns));
        }

        Matrix{
            rows,
            columns,
            data_container: weighting_vec,
        }
    }

    pub fn from_1d_vec(source : &[f64], is_vertical: bool) -> Matrix{
        let (mut rows, mut columns) = (1,1);
        let data_container = if is_vertical {
            rows = source.len();
            math::transpose_matrix(&[source.to_owned()])
        } else {
            columns = source.len();
            vec![source.to_owned()]
        };

        Matrix{
            rows,
            columns,
            data_container
        }
    }

    pub fn from_2d_vec(source: &[Vec<f64>]) -> Matrix {
        let row_size = match source.get(0) {
            Some(val) => val.len(),
            None => panic!("Cannot create a matrix from 2d vec without at least one row."),
        };
        for row in source {
            if row_size != row.len() {
                panic!("Not all rows have the same amount of columns: \
                first row -> {} columns, current row -> {} columns", row_size, row.len());
            }
        }
        Matrix {
            rows: source.len(),
            columns: row_size,
            data_container: source.to_vec(),
        }
    }

    pub fn multiply(&self, right: &Matrix) -> Result<Matrix,error::MathError> {
        let result = match math::multiply_matrices(&self.data_container, &right.data_container){
            Ok(val) => {
                Ok(Matrix {
                    rows: self.rows,
                    columns: right.columns,
                    data_container: val,
                })
            },
            Err(e) => {
                eprintln!("Left: {} rows {} cols | Right: {} rows {} cols"
                          , self.rows, self.columns, right.rows, right.columns);
                Err(e)
            }
        };

        result
    }

    pub fn add(&self, right: &Matrix) -> Result<Matrix,error::MathError> {
        let result = math::sum_matrices(&self.data_container, &right.data_container)?;

        Ok(Matrix {
            rows: right.rows,
            columns: right.columns,
            data_container: result,
        })
    }

    pub fn transpose(&self) -> Matrix {
        Matrix{
            rows: self.columns,
            columns: self.rows,
            data_container: math::transpose_matrix(&self.data_container),
        }
    }

    pub fn data_container(&self) -> &Vec<Vec<f64>>{
        &self.data_container
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
