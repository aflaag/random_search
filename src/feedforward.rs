extern crate nalgebra as na;

use crate::{activation_function::ActivationFunction, dataset::DatasetNxN};

use std::{error, fmt, ops};
use na::DMatrix;
use rand::{Rng, distributions::Standard};

pub const STEP_SIZE: f32 = 1e-5;

pub type FeedForward1x1<const S: usize> = FeedForwardNxN<1, S>;

#[derive(Debug, Clone)]
pub struct FeedForwardNxN<const N: usize, const S: usize> {
    layers_sizes: Vec<usize>,
    layers_number: usize,
    weights: Vec<DMatrix<f32>>,
    biases: Vec<DMatrix<f32>>,
    activation_function: ActivationFunction,
    dataset: DatasetNxN<f32, N, S>,
}

impl<const N: usize, const S: usize> FeedForwardNxN<N, S> {
    fn generate_matrix_from_iterator<R: Rng + ?Sized>(m: usize, n: usize, step_size: f32, rng: &mut R) -> DMatrix<f32> {
        DMatrix::from_iterator(m, n,
            (0..m * n)
                .zip(rng.sample_iter(Standard))
                .map(|(_, random): (usize, f32)| random * step_size)
        )
    }

    pub fn new<R: Rng + ?Sized>(rng: &mut R, layers_sizes: Vec<usize>, activation_function: ActivationFunction, step_size: f32) -> Result<Self, NetworkError> {
        let layers_sizes_len = layers_sizes.len();

        if layers_sizes_len == 0 {
            return Err(NetworkError::EmptyLayers);
        }

        let first_hidden_size = layers_sizes[0];
        let last_hidden_size = *layers_sizes.iter().last().unwrap(); // this unwrap is always safe

        // add the weights between the input
        // layer to the first hidden layer
        let mut weights = vec![
            Self::generate_matrix_from_iterator(first_hidden_size, N, step_size, rng)
        ];

        // add the weights bewteen the
        // hidden layers
        (0..layers_sizes_len - 1).for_each(|i| {
            let m = layers_sizes[i + 1];
            let n = layers_sizes[i];

            weights.push(Self::generate_matrix_from_iterator(m, n, step_size, rng));
        });

        // add the weights between the
        // last hidden layer and the output layer
        weights.push(Self::generate_matrix_from_iterator(N, last_hidden_size, step_size, rng));

        // create the biases for each
        // hidden layer
        let mut biases: Vec<DMatrix<f32>> = layers_sizes.iter().map(|n|
            Self::generate_matrix_from_iterator(*n, 1, step_size, rng)
        ).collect();

        // create the biases for the
        // output layer
        biases.push(Self::generate_matrix_from_iterator(N, 1, step_size, rng));

        Ok(Self {
            layers_sizes,
            layers_number: layers_sizes_len,
            weights,
            biases,
            activation_function,
            dataset: DatasetNxN::default(),
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
                output
                    .iter()
                    .zip(self.evaluate(*input).iter())
                    .map(|(o, y)| (o - y) * (o - y))
                    .fold(0.0f32, |acc, diff| acc + diff)
            })
            .fold(0.0f32, |total, cost| total + cost) / (S as f32)
    }

    pub fn random_search<R: Rng + ?Sized>(&mut self, rng: &mut R, epochs: usize, samples: usize) {
        for _ in 0..epochs {
            let best_ffnn = (0..samples).map(|_| {
                let random_ffnn = FeedForwardNxN::<N, S>::new(
                    rng,
                    self.layers_sizes.clone(),
                    self.activation_function,
                    STEP_SIZE,
                ).unwrap();

                random_ffnn
            }).min_by(|x, y|
                x
                    .evaluate_average_cost()
                    .partial_cmp(&y.evaluate_average_cost())
                    .unwrap()
            ).unwrap();

            *self += best_ffnn;
        }
    }

    pub fn evaluate(&self, input: [f32; N]) -> DMatrix<f32> {
        let mut x_vec = DMatrix::from_iterator(N, 1, input);

        // since layers_number is the number of
        // hidden layers, and we want to perform
        // the transformations with the activation
        // function only on the hidden layers, the
        // iteration from 0 to layers_number (- 1)
        // will avoid performing the loop on the
        // last layer, which is the output layer
        for i in 0..self.layers_number {
            x_vec = &self.weights[i] * x_vec + &self.biases[i];

            x_vec.iter_mut().for_each(|x| *x = (self.activation_function.function())(*x));
        }

        // in fact, here we index the weights
        // and the biases at layers_number exactly,
        // because the length of those vectors are
        // layers_number + 1, since layer_number
        // is just the number of hidden layers
        x_vec = &self.weights[self.layers_number] * x_vec + &self.biases[self.layers_number];

        x_vec
    }

    // TODO: debug
    pub fn print_stuff(&self) {
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