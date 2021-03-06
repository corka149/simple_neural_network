extern crate rand;

pub mod util;
pub mod matrix;
pub mod mnist_data;


use matrix::Matrix;
use matrix::math;
use matrix::error::*;

pub struct NeuralNetwork<T>
where
    T: Fn(f64) -> f64,
{
    learning_rate: f64,
    activation_function: T,

    wih: Matrix, // weighting: input -> hidden
    who: Matrix, // weighting: hidden -> output
}

impl<T> NeuralNetwork<T>
where
    T: Fn(f64) -> f64,
{
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f64,
        activation_function: T,
    ) -> NeuralNetwork<T> {
        let wih = Matrix::create_weighting_matrix(input_nodes, hidden_nodes);
        let who = Matrix::create_weighting_matrix(hidden_nodes, output_nodes);

        NeuralNetwork {
            learning_rate,
            activation_function,

            wih, // weighting: input -> hidden
            who, // weighting: hidden -> output
        }
    }

    pub fn train(&mut self, inputs: &[f64], awaited_output: &[f64]) -> Result<(), MathError> {
        let hidden_result = self.calculate_layer_output(inputs, self.wih.data_container());
        let final_result = self.calculate_layer_output(&hidden_result, self.who.data_container());
        let output_error = math::subtract_vectors(awaited_output, &final_result);

        let who_adjustment = self.calculate_weighting_adjustment(
            &output_error,
            &final_result,
            &hidden_result,
        )?;
        self.who = self.who.add(&Matrix::from_2d_vec(&who_adjustment))?;

        let hidden_error = match self.who.transpose().multiply(&Matrix::from_1d_vec(
            &output_error,
            true,
        )) {
            Ok(r) => r,
            Err(e) => panic!("error: {}", e),
        };

        // change from two dimensional to one dimensional matrix - Need first row
        let hidden_error = math::transpose_2d_vector(hidden_error.data_container());
        let hidden_error = match hidden_error.first() {
            Some(he) => he,
            None => panic!("hidden error can never be without a first row!!!"),
        };

        let wih_adjustment = self.calculate_weighting_adjustment(
            hidden_error,
            &hidden_result,
            inputs,
        )?;
        self.wih = self.wih.add(&Matrix::from_2d_vec(&wih_adjustment))?;

        Ok(())
    }

    pub fn query(&self, inputs: &[f64]) -> Vec<f64> {
        let hidden_outputs = self.calculate_layer_output(inputs, self.wih.data_container());
        self.calculate_layer_output(&hidden_outputs, self.who.data_container())
    }

    fn calculate_layer_output(&self, inputs: &[f64], weighting: &[Vec<f64>]) -> Vec<f64> {
        let inputs = math::transpose_2d_vector(&[inputs.to_owned()]);
        let weighted_input = match math::multiply_matrices(weighting, &inputs) {
            Ok(r) => r,
            Err(e) => panic!("{}", e),
        };

        let mut outputs: Vec<f64> = Vec::new();
        for row in weighted_input {
            // row.iter().sum() because there is only on value per row #shortcut
            outputs.push((self.activation_function)(row.iter().sum()));
        }
        outputs
    }

    fn calculate_weighting_adjustment(
        &self,
        error: &[f64],
        output: &[f64],
        previous_output: &[f64],
    ) -> Result<Vec<Vec<f64>>, MathError> {
        let inner_result: Vec<f64> = error
            .iter()
            .zip(output)
            .map(|(x, y)| x * y * (1.0 - y))
            .collect();
        let inner_result = math::transpose_2d_vector(&[inner_result]);
        let mut outer_result =
            math::multiply_matrices(&inner_result, &[previous_output.to_owned()])?;

        for row in &mut outer_result {
            for cell in row.iter_mut() {
                *cell *= self.learning_rate;
            }
        }

        Ok(outer_result)
    }
}

#[cfg(test)]
mod neural_network_tests {
    use super::*;

    #[test]
    fn create_new_neural_network() {
        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);

        assert_eq!(nn.input_nodes, 3);
        assert_eq!(nn.hidden_nodes, 3);
        assert_eq!(nn.output_nodes, 3);
        assert_eq!(nn.learning_rate, 0.3);
        assert_eq!((nn.activation_function)(1.0), 2.0);
    }

    // Running train() should never panic
    #[test]
    fn test_train() {
        let mut nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);

        let inputs: Vec<f64> = vec![1.0, 1.0, 1.0];
        let outputs: Vec<f64> = vec![1.0, 1.0, 1.0];

        nn.train(&inputs, &outputs);
    }

    // Running query() should never panic
    #[test]
    fn test_query() {
        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);

        let inputs: Vec<f64> = vec![1.0, 1.0, 1.0];

        nn.query(&inputs);
    }

    #[test]
    fn test_calculate_weighting_adjustment() {
        let nn = NeuralNetwork::new(2, 2, 2, 0.5, |x| x + 1.0);

        let err = vec![0.2, 0.15];
        let fin_result = vec![0.9, 0.7];
        let hidden_result = vec![0.5, 0.8];

        let result = nn.calculate_weighting_adjustment(&err, &fin_result, &hidden_result)
            .unwrap();

        assert_eq!(result[0][0], 0.0045);
        assert_eq!(result[0][1], 0.0072);
        assert_eq!(result[1][0], 0.007875);
        assert_eq!(result[1][1], 0.0126);
    }
}
