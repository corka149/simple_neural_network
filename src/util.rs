use rand;
use rand::Rng;
use std::f64::consts::E;

pub fn create_weighting_vec(x: u64, y: u64) -> Vec<Vec<f64>> {
    let mut weighting_vec = Vec::new();

    for number in 0..y {
        weighting_vec.push(create_weighting_row(x));
    }
    weighting_vec
}

pub fn create_weighting_row(x: u64) -> Vec<f64> {
    let mut row: Vec<f64> = Vec::new();

    for number in 0..x {
        row.push(rand::thread_rng().gen_range(-0.5, 0.5));
    }
    row
}

pub fn sigmoid(x: f64) -> f64 {
    (1.0 / (1.0 + E.powf(-x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_weighting_row() {
        let row = create_weighting_row(3);
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn test_create_weighting_vec() {
        let vec = create_weighting_vec(3, 4);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0].len(), 3);
    }

    #[test]
    fn test_sigmoid() {
        let y = 0.8807970779778823;
        let x = sigmoid(2.0);
        assert_eq!(y, x);
    }
}
