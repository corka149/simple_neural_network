extern crate rand;

extern crate matrix;

pub mod util;
pub mod mnist_data;
pub mod math;
pub mod error;

use matrix::prelude::*;
use error::MathError;

pub struct NeuralNetwork<T>
where
    T: Fn(f64) -> f64,
{
    learning_rate: f64,
    activation_function: T,

    wih: Compressed<f64>, // weighting: input -> hidden
    who: Compressed<f64>, // weighting: hidden -> output
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
        let wih = util::create_weighting_matrix(input_nodes, hidden_nodes);
        let who = util::create_weighting_matrix(hidden_nodes, output_nodes);

        NeuralNetwork {
            learning_rate,
            activation_function,

            wih, // weighting: input -> hidden
            who, // weighting: hidden -> output
        }
    }

    pub fn train(
        &mut self,
        inputs: &Compressed<f64>,
        awaited_output: &Compressed<f64>,
    ) -> Result<(), MathError> {
        let hidden_result = self.calculate_layer_output(inputs, &self.wih);
        let final_result = self.calculate_layer_output(&hidden_result, &self.who);
        let output_error = match math::subtract_matrices(awaited_output, &final_result){
            Ok(val) => val,
            Err(e) => panic!("{}", e),
        };

        let who_adjustment = self.calculate_weighting_adjustment(
            &output_error,
            &final_result,
            &hidden_result,
        )?;
        self.who = math::sum_matrices(&self.who, &who_adjustment)?;

        let hidden_error = self.who.multiply(&output_error);

        let wih_adjustment = self.calculate_weighting_adjustment(
            &hidden_error,
            &hidden_result,
            inputs,
        )?;
        self.wih = math::sum_matrices(&self.wih, &wih_adjustment)?;

        Ok(())
    }

    pub fn query(&self, inputs: &Compressed<f64>) -> Compressed<f64> {
        let hidden_outputs = self.calculate_layer_output(inputs, &self.wih);
        self.calculate_layer_output(&hidden_outputs, &self.who)
    }

    fn calculate_layer_output(&self, inputs: &Compressed<f64>, weighting: &Compressed<f64>) -> Compressed<f64> {
        let weighted_input = weighting.multiply(inputs);

        let mut outputs = Compressed::zero((weighted_input.rows,1));
        for x in 0..weighted_input.rows {
            for y in 0..weighted_input.columns{
                let val = weighted_input.get((x,y));
                outputs.set((x,y), (self.activation_function)(val));
            }
        }
        outputs
    }

    fn calculate_weighting_adjustment(
        &self,
        error: &Compressed<f64>,
        output: &Compressed<f64>,
        previous_output: &Compressed<f64>,
    ) -> Result<Compressed<f64>, MathError> {
        let mut inner_result: Compressed<f64> = Compressed::zero((error.rows,1));
        for x in 0..error.rows {
            for y in 0..error.columns {
                let pos = (x,y);
                let e_val = error.get(pos);
                let o_val = output.get(pos);
                let final_val = e_val * o_val * (1.0 - o_val);
                inner_result.set(pos, final_val);
            }
        }

        let mut outer_result = inner_result.multiply(previous_output);
        for x in 0..outer_result.rows {
            for y in 0..outer_result.columns {
                let pos = (x,y);
                outer_result.set(pos, outer_result.get(pos) * self.learning_rate);
            }
        }

        Ok(outer_result)
    }
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    #[test]
//    fn create_new_neural_network() {
//        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);
//
//        assert_eq!(nn.input_nodes, 3);
//        assert_eq!(nn.hidden_nodes, 3);
//        assert_eq!(nn.output_nodes, 3);
//        assert_eq!(nn.learning_rate, 0.3);
//        assert_eq!((nn.activation_function)(1.0), 2.0);
//    }
//
//    #[test]
//    fn test_train() {
//        let mut nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);
//
//        let inputs: Vec<f64> = vec![1.0, 1.0, 1.0];
//        let outputs: Vec<f64> = vec![1.0, 1.0, 1.0];
//
//        nn.train(&inputs, &outputs);
//    }
//
//    #[test]
//    fn test_query() {
//        let nn = NeuralNetwork::new(3, 3, 3, 0.3, |x| x + 1.0);
//
//        let inputs: Vec<f64> = vec![1.0, 1.0, 1.0];
//
//        nn.query(&inputs);
//    }
//
//    #[test]
//    fn test_calculate_weighting_adjustment() {
//        let nn = NeuralNetwork::new(2, 2, 2, 0.5, |x| x + 1.0);
//
//        let err = vec![0.2, 0.15];
//        let fin_result = vec![0.9, 0.7];
//        let hidden_result = vec![0.5, 0.8];
//
//        let result = nn.calculate_weighting_adjustment(&err, &fin_result, &hidden_result)
//            .unwrap();
//
//        assert_eq!(result[0][0], 0.0045);
//        assert_eq!(result[0][1], 0.0072);
//        assert_eq!(result[1][0], 0.007875);
//        assert_eq!(result[1][1], 0.0126);
//    }
//}
