extern crate rand;

pub mod util;

struct NeuralNetwork<T>
    where T: Fn(i64) -> i64
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
    where T: Fn(i64) -> i64
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
            wih,
            who,
        }
    }

    pub fn train(&self, inputs: &[i8], output: &[i8]) {}

    pub fn query(&self, inputs: &[i8]) -> Vec<i8> {
        vec![]
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn create_new_neural_network() {
        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1);

        assert_eq!(nn.input_nodes, 3);
        assert_eq!(nn.hidden_nodes, 3);
        assert_eq!(nn.output_nodes, 3);
        assert_eq!(nn.learning_rate, 0.3);
        assert_eq!((nn.activation_function)(1), 2);
    }

    #[test]
    fn test_query() {
        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1);

        let inputs: Vec<i8> = vec![1];
        let outputs: Vec<i8> = vec![1];

        nn.train(&inputs, &outputs);
    }
}
