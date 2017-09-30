extern crate rand;

pub mod util;
pub mod vector;

use vector::math;

pub struct NeuralNetwork<T>
    where T: Fn(f64) -> f64
{
    input_nodes: u64,
    hidden_nodes: u64,
    output_nodes: u64,
    learning_rate: f64,

    activation_function: T,

    wih: Vec<Vec<f64>>,
    who: Vec<Vec<f64>>,
}

impl<T> NeuralNetwork<T>
    where T: Fn(f64) -> f64
{
    pub fn new(input_nodes: u64,
               hidden_nodes: u64,
               output_nodes: u64,
               learning_rate: f64,
               activation_function: T)
               -> NeuralNetwork<T> {
        let wih = util::create_weighting_vec(input_nodes, hidden_nodes);
        let who = util::create_weighting_vec(hidden_nodes, output_nodes);

        NeuralNetwork {
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            activation_function,
            wih, // weighting: input -> hidden
            who, // weighting: hidden -> output
        }
    }

    pub fn train(&mut self,
                 inputs: &Vec<f64>,
                 awaited_output: &Vec<f64>)
                 -> Result<(), vector::MathError> {
        let final_result = self.query(inputs);
        let hidden_result = self.calculate_layer_output(inputs, &self.wih);
        let output_error = math::subtract_simple_vectors(&final_result, &awaited_output);

        let who_adjustment =
            self.calculate_weighting_adjustment(&output_error, &final_result, &hidden_result)?;
        self.who = math::sum_vectors(&self.who, &who_adjustment)?;

        let hidden_error =
            match math::multiply_vectors(&self.who, &math::transpose_vec(&vec![output_error])) {
                Ok(r) => r,
                Err(e) => panic!("{}", e),
            };

        // change from two dimensional to one dimensional vector
        let hidden_error = math::transpose_vec(&hidden_error);
        let hidden_error = match hidden_error.first() {
            Some(he) => he,
            None => panic!("hidden error can never be without a first row!!!"),
        };

        let wih_adjustment =
            self.calculate_weighting_adjustment(&hidden_error, &hidden_result, inputs)?;
        self.wih = math::sum_vectors(&self.wih, &wih_adjustment)?;

        Ok(())
    }

    pub fn query(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let hidden_inputs = self.calculate_layer_output(&inputs, &self.wih);
        self.calculate_layer_output(&hidden_inputs, &self.who)
    }

    fn calculate_layer_output(&self, inputs: &Vec<f64>, weighting: &Vec<Vec<f64>>) -> Vec<f64> {
        let inputs = math::transpose_vec(&vec![inputs.clone()]);
        let weighted_input = match math::multiply_vectors(weighting, &inputs) {
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

    fn calculate_weighting_adjustment(&self,
                                      error: &Vec<f64>,
                                      output: &Vec<f64>,
                                      previous_output: &Vec<f64>)
                                      -> Result<Vec<Vec<f64>>, vector::MathError> {
        let inner_result: Vec<f64> = error
            .iter()
            .zip(output)
            .map(|(x, y)| x * y * (1.0 - y))
            .collect();
        let inner_result = math::transpose_vec(&vec![inner_result]);
        let mut outer_result = math::multiply_vectors(&inner_result,
                                                      &vec![previous_output.clone()])?;

        for row in outer_result.iter_mut() {
            for cell in row.iter_mut() {
                *cell *= self.learning_rate;
            }
        }

        Ok(outer_result)
    }
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_train() {
        let mut nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);

        let inputs: Vec<f64> = vec![1.0, 1.0, 1.0];
        let outputs: Vec<f64> = vec![1.0, 1.0, 1.0];

        nn.train(&inputs, &outputs);
    }

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
