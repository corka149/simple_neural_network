use std::error;
use std::fmt;

pub mod math;

#[derive(Debug,Clone)]
pub struct MathError;

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "incompatible vectors: Sizes does not match")
    }
}

impl error::Error for MathError {
    fn description(&self) -> &str {
        "incompatible vectors: Sizes does not match"
    }

    fn cause(&self) -> Option<&error::Error> {
        None
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
