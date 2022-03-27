extern crate nalgebra as na;

use crate::{activation_function::ActivationFunction, dataset::DatasetNxN};

use na::DMatrix;
use rand::{Rng, thread_rng};

pub struct FeedForwardNxN<const L: usize, const S: usize> {
    // layers_sizes: Vec<usize>, // TODO: check if this is useful, actually also the other ones
    layers_number: usize,
    weights: Vec<DMatrix<f32>>,
    biases: Vec<DMatrix<f32>>,
    activation_function: fn(f32) -> f32,
    dataset: DatasetNxN<f32, L, S>,
}

impl<const L: usize, const S: usize> FeedForwardNxN<L, S> {
    pub fn new(layers_sizes: Vec<usize>, activation_function: ActivationFunction, dataset: DatasetNxN<f32, L, S>) -> Self {
        let mut rng = thread_rng();

        // TODO: change this shit with an enum for the errors
        let first_hidden_size = *layers_sizes.get(0).expect("There must be at least 1 hidden layer");
        let last_hidden_size = *layers_sizes.iter().last().expect("There must be at least 1 hidden layer");

        // add the weights from the input
        // layer to the first hidden layer
        let mut weights = vec![
            DMatrix::<f32>::from_vec(first_hidden_size, L, (0..first_hidden_size * L).map(|_| rng.gen()).collect())
        ];

        // add the weights bewteen the
        // hidden layers
        (0..layers_sizes.len() - 1).for_each(|i| {
            let m = layers_sizes[i + 1];
            let n = layers_sizes[i];

            weights.push(DMatrix::from_vec(m, n, (0..m * n).map(|_| rng.gen()).collect()));
        });

        // add the weights between the
        // last hidden layer and the output layer
        weights.push(
            DMatrix::<f32>::from_vec(L, last_hidden_size, (0..last_hidden_size * L).map(|_| rng.gen()).collect())
        );

        // create the biases for each
        // hidden layer
        let mut biases: Vec<DMatrix<f32>> = layers_sizes.iter().map(|n|
            DMatrix::from_vec(*n, 1, (0..*n).map(|_| rng.gen()).collect())
        ).collect();

        // create the biases for the
        // output layer
        biases.push(DMatrix::from_vec(L, 1, (0..L).map(|_| rng.gen()).collect()));

        Self {
            layers_number: layers_sizes.len(),
            weights,
            biases,
            activation_function: activation_function.value(),
            dataset,
        }
    }

    pub fn train(&mut self) {
        self.dataset.get_input().iter().zip(self.dataset.get_output()).for_each(|(input, output)| {
            let mut x_vec = DMatrix::from_iterator(L, 1, *input);

            // since layers_number is the number of
            // hidden layers, and we want to perform
            // the transformations with the activation
            // function only on the hidden layers, the
            // iteration from 0 to layers_number (- 1)
            // will avoid performing the loop on the
            // last layer, which is the output layer
            for i in 0..self.layers_number {
                x_vec = self.weights[i].clone() * x_vec + self.biases[i].clone();

                x_vec.iter_mut().for_each(|x| *x = (self.activation_function)(*x));
            }

            // in fact, here we index the weights
            // and the biases at layers_number exactly,
            // because the length of those vectors are
            // layers_number + 1, since layer_number
            // is just the number of hidden layers
            x_vec = self.weights[self.layers_number].clone() * x_vec + self.biases[self.layers_number].clone();

            let cost = output
                .iter()
                .zip(x_vec.iter())
                .map(|(o, y)| (o - y) * (o - y))
                .fold(0.0f32, |acc, c| acc + c);

            println!("{}", cost); // TODO: debug
        });

        // TODO: INCOMPLETE
    }

    // TODO: debug
    pub fn print_stuff(&self) {
        println!("{:#?}", self.weights);
        println!("{:#?}", self.biases);
    }
}