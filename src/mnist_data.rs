/// Converts one data set.
pub fn convert_mnist_line(line: &str) -> (usize, Vec<f64>) {
    // Is a iterator
    let elements = line.split(',');
    let values: Vec<f64> = elements
        .map(|x: &str| match x.trim().parse() {
            Ok(v) => v,
            Err(e) => panic!("{:?}", e),
        })
        .collect();

    let specific_number = match values.first() {
        Some(v) => *v as usize,
        None => panic!("First value in mnist_line is not a number"),
    };

    let values: Vec<f64> = values
        .iter()
        .skip(1)
        .map(|x| x / 255.0 * 0.99 + 0.01)
        .collect();

    (specific_number, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_mnist_line() {
        let test_line = "1,255,16,100";
        let (number, values) = convert_mnist_line(test_line);

        assert_eq!(number, 1);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.07211764705882352);
        assert_eq!(values[2], 0.3982352941176471);
    }

}
