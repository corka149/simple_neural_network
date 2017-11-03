use super::*;
use super::error::*;

pub fn subtract_matrices(minuend: &Compressed<f64>, subtrahend: &Compressed<f64>) -> Result<Compressed<f64>, MathError> {
    let result: Result<Compressed<f64>, MathError>;

    if minuend.columns != subtrahend.columns || minuend.rows != subtrahend.rows {
        result = Err(MathError);
    } else {
        let mut difference: Compressed<f64> = Compressed::zero((minuend.rows, minuend.columns));
        for x in 0..minuend.rows {
            for y in 0..minuend.columns {
                let pos = (x, y);
                difference.set(pos, minuend.get(pos) - subtrahend.get(pos));
            }
        }
        result = Ok(difference);
    }
    result
}

pub fn sum_matrices(first: &Compressed<f64>,
                    second: &Compressed<f64>)
                    -> Result<Compressed<f64>, MathError> {
    let result: Result<Compressed<f64>, MathError>;

    if first.rows != second.rows || first.columns != second.columns {
        result = Err(MathError);
    } else {
        let mut sum: Compressed<f64> = Compressed::zero((first.rows,first.columns));
        for x in 0..first.rows {
            for y in 0..first.columns {
                let pos = (x,y);
                sum.set(pos, first.get(pos) + second.get(pos));
            }
        }
        result = Ok(sum);
    }
    result
}


//#[cfg(test)]
//mod tests {
//    use super::*;
//
//
//    #[test]
//    fn test_subtract_matrices() {
//        let vec_a = vec![2.0, 3.0];
//        let vec_b = vec![1.0, 1.5];
//        let result = subtract_matrices(&vec_a, &vec_b);
//        assert_eq!(result[0], 1.0);
//        assert_eq!(result[1], 1.5);
//    }
//
//    #[test]
//    fn test_sum_matrices() {
//        let first = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
//        let second = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
//        let result = sum_matrices(&first, &second).unwrap();
//        assert_eq!(result[0][0], 6.0);
//        assert_eq!(result[0][1], 8.0);
//        assert_eq!(result[1][0], 10.0);
//        assert_eq!(result[1][1], 12.0);
//    }
//}
