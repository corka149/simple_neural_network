use rand;
use rand::Rng;
use std::f64::consts::E;
use matrix::prelude::*;

pub fn create_weighting_matrix(x: usize, y: usize) -> Compressed<f64> {
    let mut weightings: Compressed<f64> = Compressed::zero((x, y));

    for _number in 0..x {
        for _number in 0..y {
            let min = rand::thread_rng().gen_range(-0.1, -0.01);
            let max = rand::thread_rng().gen_range(0.01, 0.1);
            let val = rand::thread_rng().gen_range(min, max);
            weightings.set((x, y), val);
        }
    }

    weightings
}

pub fn sigmoid(x: f64) -> f64 {
    (1.0 / (1.0 + E.powf(-x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_weighting_matrix() {
        let matrix = create_weighting_matrix(3, 4);
        let mut too_high = Vec::new();

        for _x in 0..matrix.rows {
            for _y in 0..matrix.columns {
                let val = matrix.get((x, y));
                if val < -0.5 && val > 0.5 {
                    too_high.push(val);
                }
            }
        }

        assert_eq!(matrix.rows, 4);
        assert_eq!(matrix.columns, 3);
    }

    #[test]
    fn test_sigmoid() {
        let y = 0.8807970779778823;
        let x = sigmoid(2.0);
        assert_eq!(y, x);
    }
}
