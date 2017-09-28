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
