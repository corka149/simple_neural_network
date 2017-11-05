extern crate neural_network;
extern crate chrono;

use neural_network::*;
use std::fs::File;
use std::io::prelude::*;
use chrono::prelude::*;

fn main() {
    let mut wrong = 0;
    let mut correct = 0;
    let mut nn = NeuralNetwork::new(784,200,10,0.1, util::sigmoid);

    let mut awaited_output: Vec<f64>;

    println!("Start time: {}", Local::now());

    for (index, line) in unpack("train.csv").lines().enumerate() {
        let (number, values) = mnist_data::convert_mnist_line(line);
        awaited_output = matrix::create_zeroed_vector(10);
        match awaited_output.get_mut(number) {
            Some(v) => *v = 0.99,
            None => panic!("number {} could not occur!", number),
        }
        if index % 600 == 0 {
            println!("{} %", index / 600);
        }
        if let Err(e) = nn.train(&values, &awaited_output) {
            eprintln!("{}", e);
        };
    }

    // ----------------------------------------------------------

    for line in unpack("test.csv").lines() {
        let (number, values) = mnist_data::convert_mnist_line(line);
        let mut answer: usize = 0;
        let mut highest: f64 = 0.0;

        for (index, value) in nn.query(&values).iter().enumerate() {
            if *value > highest {
                highest = *value;
                answer = index;
            }
        }
        if number == answer {
            correct += 1;
        }else{
            wrong += 1;
        }
    }
    println!("Correct answers {}", correct);
    println!("Finish! Error rate {} %", f64::from(wrong) / (f64::from(wrong) + f64::from(correct)) * 100.0);
    println!("End time: {}", Local::now());
}

fn unpack(file_name: &str) -> String {
    let mut file = match File::open(file_name) {
        Ok(f) => f,
        Err(e) => panic!("{}", e),
    };

    let mut content = String::new();
    if let Err(e) = file.read_to_string(&mut content){
        panic!("{}", e);
    }
    content
}
