extern crate rand;

pub mod util;
pub mod vector;

use vector::math;

struct NeuralNetwork<T>
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
            wih, // input -> hidden
            who, // hidden -> output
        }
    }

    pub fn train(&self, inputs: Vec<f64>, output: &Vec<f64>) {}

    pub fn query(&self, inputs: Vec<f64>) -> Vec<f64> {
        let hidden_inputs = self.calculate_layer_output(inputs, &self.wih);
        self.calculate_layer_output(hidden_inputs, &self.who)
    }

    fn calculate_layer_output(&self, inputs: Vec<f64>, weighting: &Vec<Vec<f64>>) -> Vec<f64> {
        let inputs = math::transpose_vec(&vec![inputs]);
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
        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);

        let inputs: Vec<f64> = vec![1.0];
        let outputs: Vec<f64> = vec![1.0];

        nn.train(inputs, &outputs);
    }

    #[test]
    fn test_query() {
        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);

        let inputs: Vec<f64> = vec![1.0,1.0, 1.0];

        nn.query(inputs);
    }
}
