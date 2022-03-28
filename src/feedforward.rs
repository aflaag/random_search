extern crate nalgebra as na;

use crate::{activation_function::ActivationFunction, dataset::DatasetNxN};

use std::{error, fmt, ops};
use na::DMatrix;
use rand::{Rng, thread_rng, distributions::Standard};

const STEP_SIZE: f32 = 1e-4;

pub type FeedForward1x1<const S: usize> = FeedForwardNxN<1, S>;

#[derive(Debug, Clone)]
pub struct FeedForwardNxN<const N: usize, const S: usize> {
    layers_sizes: Vec<usize>,
    layers_number: usize,
    weights: Vec<DMatrix<f32>>,
    biases: Vec<DMatrix<f32>>,
    activation_function: ActivationFunction,
    dataset: DatasetNxN<f32, N, S>,
    // size: usize,
}

impl<const N: usize, const S: usize> FeedForwardNxN<N, S> {
    fn generate_matrix_from_iterator<R: Rng + Clone>(m: usize, n: usize, rng: R) -> DMatrix<f32> {
        DMatrix::from_iterator(m, n,
            (0..m * n)
                .zip(rng.sample_iter(Standard))
                .map(|(_, random): (usize, f32)| random * STEP_SIZE)
        )
    }

    pub fn new<R: Rng + Clone>(
        rng: &mut R,
        layers_sizes:
        Vec<usize>,
        activation_function:
        ActivationFunction,
    ) -> Result<Self, NetworkError> {
        let layers_sizes_len = layers_sizes.len();

        if layers_sizes_len == 0 {
            return Err(NetworkError::EmptyLayers);
        }

        let first_hidden_size = layers_sizes[0];
        let last_hidden_size = *layers_sizes.iter().last().unwrap(); // this unwrap is always safe

        // add the weights between the input
        // layer to the first hidden layer
        let mut weights = vec![
            Self::generate_matrix_from_iterator(first_hidden_size, N, rng.clone())
        ];

        // add the weights bewteen the
        // hidden layers
        (0..layers_sizes_len - 1).for_each(|i| {
            let m = layers_sizes[i + 1];
            let n = layers_sizes[i];

            weights.push(Self::generate_matrix_from_iterator(m, n, rng.clone()));
        });

        // add the weights between the
        // last hidden layer and the output layer
        weights.push(Self::generate_matrix_from_iterator(N, last_hidden_size, rng.clone()));

        // create the biases for each
        // hidden layer
        let mut biases: Vec<DMatrix<f32>> = layers_sizes.iter().map(|n|
            Self::generate_matrix_from_iterator(*n, 1, rng.clone())
        ).collect();

        // create the biases for the
        // output layer
        biases.push(Self::generate_matrix_from_iterator(N, 1, rng.clone()));

        Ok(Self {
            layers_sizes,
            layers_number: layers_sizes_len,
            weights,
            biases,
            activation_function: activation_function,
            dataset: DatasetNxN::default(),
            // size:
            //     layers_sizes.iter().sum::<usize>() + N + // biases hidden layers + bias output layer
            //     N * first_hidden_size + // weights between the input layer and the first hidden layer
            //     layers_sizes.windows(2).map(|w| w[0] * w[1]).sum::<usize>() + // weights between hidden layers
            //     N * last_hidden_size // weights between the last hidden layer and the output layer
        })
    }

    pub fn load_dataset(&mut self, dataset: DatasetNxN<f32, N, S>) {
        self.dataset = dataset;
    }

    fn evaluate_average_cost(&self) -> f32 {
        self.dataset
            .get_input()
            .iter()
            .zip(self.dataset.get_output())
            .map(|(input, output)| {
                let mut x_vec = DMatrix::from_iterator(N, 1, *input);

                // since layers_number is the number of
                // hidden layers, and we want to perform
                // the transformations with the activation
                // function only on the hidden layers, the
                // iteration from 0 to layers_number (- 1)
                // will avoid performing the loop on the
                // last layer, which is the output layer
                for i in 0..self.layers_number {
                    x_vec = self.weights[i].clone() * x_vec + self.biases[i].clone();

                    x_vec.iter_mut().for_each(|x| *x = (self.activation_function.value())(*x));
                }

                // in fact, here we index the weights
                // and the biases at layers_number exactly,
                // because the length of those vectors are
                // layers_number + 1, since layer_number
                // is just the number of hidden layers
                x_vec = self.weights[self.layers_number].clone() * x_vec + self.biases[self.layers_number].clone();

                output
                    .iter()
                    .zip(x_vec.iter())
                    .map(|(o, y)| (o - y) * (o - y))
                    .fold(0.0f32, |acc, diff| acc + diff)
            })
            .fold(0.0f32, |total, cost| total + cost) / S as f32
    }

    pub fn random_search<R: Rng + Clone>(&mut self, rng: &mut R) {
        let average_cost = self.evaluate_average_cost();

        let random_ffnn = FeedForwardNxN::<N, S>::new(
            rng,
            self.layers_sizes.clone(),
            self.activation_function,
        ).unwrap();

        *self += random_ffnn;

        println!("{}", average_cost);
    }

    // TODO: debug
    pub fn print_stuff(&self) {
        // println!("{:#?}", self.weights);
        // println!("{:#?}", self.biases);
        // println!("{:#?}", self.size);
    }
}

impl<const N: usize, const S: usize> ops::AddAssign for FeedForwardNxN<N, S> {
    fn add_assign(&mut self, rhs: Self) {
        if self.layers_sizes != rhs.layers_sizes {
            panic!("The two networks have different sizes");
        } else {
            self.weights.iter_mut().zip(rhs.weights.iter()).for_each(|(self_matrix, rhs_matrix)| {
                self_matrix.iter_mut().zip(rhs_matrix.iter()).for_each(|(v, r)| {
                    *v += r;
                });
            });

            self.biases.iter_mut().zip(rhs.biases.iter()).for_each(|(self_bias, rhs_bias)| {
                self_bias.iter_mut().zip(rhs_bias.iter()).for_each(|(v, r)| {
                    *v += r;
                });
            });
        }
    }
}

#[derive(Debug)]
pub enum NetworkError {
    EmptyLayers,
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyLayers => writeln!(f, "The network must have at least 1 hidden layer."),
        }
    }
}

impl error::Error for NetworkError {}