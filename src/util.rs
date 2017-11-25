use rand;
use rand::Rng;
use std::f64::consts::E;

pub fn create_weighting_row(x: u64) -> Vec<f64> {
    let mut row: Vec<f64> = Vec::new();

    for _number in 0..x {
        let min = rand::thread_rng().gen_range(-0.1, -0.01);
        let max = rand::thread_rng().gen_range(0.01, 0.1);
        row.push(rand::thread_rng().gen_range(min, max));
    }
    row
}

pub fn sigmoid(x: f64) -> f64 {
    (1.0 / (1.0 + E.powf(-x)))
}

#[cfg(test)]
mod util_tests {
    use super::*;

    #[test]
    fn test_create_weighting_row() {
        let row = create_weighting_row(1000);
        assert_eq!(row.len(), 1000);

        let check_values: Vec<&f64> = row.iter().filter(|x| **x < -0.5 && **x > 0.5).collect();

        assert_eq!(check_values.len() , 0);
    }

    #[test]
    fn test_sigmoid() {
        let y = 0.8807970779778823;
        let x = sigmoid(2.0);
        assert_eq!(y, x);
    }
}
