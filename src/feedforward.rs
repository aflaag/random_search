extern crate nalgebra as na;

use crate::{activation_function::ActivationFunction, dataset::DatasetNxN, cost_function::CostFunction};

use std::{error, fmt, ops};
use na::DMatrix;
use rand::{Rng, prelude::StdRng, SeedableRng};
use rand_distr::{StandardNormal, Standard};
use rayon::prelude::*;

pub const STEP_SIZE: f32 = 0.0001;

pub type FeedForward1x1<const S: usize> = FeedForwardNxN<1, S>;

#[derive(Debug, Clone)]
pub struct FeedForwardNxN<const N: usize, const S: usize> {
    layers_sizes: Vec<usize>,
    layers_number: usize,
    weights: Vec<DMatrix<f32>>,
    biases: Vec<DMatrix<f32>>,
    activation_function: ActivationFunction,
    cost_function: CostFunction,
    dataset: DatasetNxN<N, S>,
}

impl<const N: usize, const S: usize> FeedForwardNxN<N, S> {
    fn generate_matrix_from_iterator<R: Rng + ?Sized>(m: usize, n: usize, rng: &mut R) -> DMatrix<f32> {
        DMatrix::from_iterator(m, n,
            rng.sample_iter::<f32, _>(StandardNormal).take(m * n).map(|random| random * STEP_SIZE)
        )
    }

    pub fn new<R: Rng + ?Sized>(
        rng: &mut R,
        layers_sizes: Vec<usize>,
        activation_function: ActivationFunction,
        cost_function: CostFunction,
        dataset: DatasetNxN<N, S>
    ) -> Result<Self, NetworkError> {
        let layers_sizes_len = layers_sizes.len();

        if layers_sizes_len == 0 {
            return Err(NetworkError::EmptyLayers);
        }

        if layers_sizes.iter().any(|size| *size == 0) {
            return Err(NetworkError::ZeroLayer)
        }

        let first_hidden_size = layers_sizes[0];
        let last_hidden_size = *layers_sizes.iter().last().unwrap(); // this unwrap is always safe

        // add the weights between the input
        // layer to the first hidden layer
        let mut weights = vec![
            Self::generate_matrix_from_iterator(first_hidden_size, N, rng)
        ];

        // add the weights bewteen the
        // hidden layers
        layers_sizes.windows(2).for_each(|window|
            weights.push(Self::generate_matrix_from_iterator(window[1], window[0], rng))
        );

        // add the weights between the
        // last hidden layer and the output layer
        weights.push(Self::generate_matrix_from_iterator(N, last_hidden_size, rng));

        // create the biases for each
        // hidden layer
        let mut biases: Vec<DMatrix<f32>> = layers_sizes.iter().map(|n|
            Self::generate_matrix_from_iterator(*n, 1, rng)
        ).collect();

        // create the biases for the
        // output layer
        biases.push(Self::generate_matrix_from_iterator(N, 1, rng));

        Ok(Self {
            layers_sizes,
            layers_number: layers_sizes_len,
            weights,
            biases,
            activation_function,
            cost_function,
            dataset,
        })
    }

    pub fn evaluate_average_cost(&self, verbose: bool) -> f32 {
        self.dataset
            .iter_zip()
            .map(|(input, expected)| {
                let output = self.evaluate(DMatrix::<f32>::from_iterator(N, 1, *input));

                if verbose {
                    println!("({:?}, {:?})", input[0], output[0]);
                }

                output.iter().zip(expected).map(|(o, y)| (self.cost_function.function())(*o, *y)).sum::<f32>()
            })
            .sum::<f32>() / (S as f32)
    }

    pub fn random_search<R: Rng + ?Sized>(&mut self, rng: &mut R, epochs: usize, networks: usize, verbose: bool) {
        for epoch in 1..=epochs {
            // generates some random seeds
            // which are then used to keep track
            // of the best network
            let lowest_seed = rng
                .sample_iter(Standard)
                .take(networks)
                .collect::<Vec<u64>>()
                .into_par_iter()
                .map(|seed| {
                    let mut seeded_rng = StdRng::seed_from_u64(seed);

                    let ffnn = FeedForwardNxN::<N, S>::new(
                        &mut seeded_rng,
                        self.layers_sizes.clone(),
                        self.activation_function,
                        self.cost_function,
                        self.dataset,
                    ).unwrap();

                    ((ffnn + self.clone()).unwrap().evaluate_average_cost(false), seed)
                })
                .min_by(|(x, _), (y, _)| x.partial_cmp(&y).unwrap())
                .map(|(_, seed)| seed)
                .unwrap();

            *self += FeedForwardNxN::<N, S>::new(
                &mut StdRng::seed_from_u64(lowest_seed),
                self.layers_sizes.clone(),
                self.activation_function,
                self.cost_function,
                self.dataset,
            ).unwrap();

            if verbose {
                println!("({}, {})", epoch, self.evaluate_average_cost(false));
            }
        }
    }

    pub fn evaluate(&self, mut x_vec: DMatrix<f32>) -> DMatrix<f32> {
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
        &self.weights[self.layers_number] * x_vec + &self.biases[self.layers_number]
    }
}

impl<const N: usize, const S: usize> ops::Add for FeedForwardNxN<N, S> {
    type Output = Result<Self, NetworkError>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.layers_sizes != rhs.layers_sizes {
            Err(NetworkError::DifferentSizes)
        } else {
            let mut result = self;

            result += rhs;
            
            Ok(result)
        }
    }
}

impl<const N: usize, const S: usize> ops::AddAssign for FeedForwardNxN<N, S> {
    fn add_assign(&mut self, rhs: Self) {
        if self.layers_sizes != rhs.layers_sizes {
            panic!("The two networks have different sizes.");
        } else {
            self.weights.iter_mut().zip(rhs.weights.iter()).for_each(|(self_matrix, rhs_matrix)| {
                self_matrix.iter_mut().zip(rhs_matrix.iter()).for_each(|(v, r)| *v += r);
            });

            self.biases.iter_mut().zip(rhs.biases.iter()).for_each(|(self_bias, rhs_bias)| {
                self_bias.iter_mut().zip(rhs_bias.iter()).for_each(|(v, r)| *v += r);
            });
        }
    }
}

#[derive(Debug)]
pub enum NetworkError {
    EmptyLayers,
    DifferentSizes,
    ZeroLayer,
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyLayers => writeln!(f, "The network must have at least 1 hidden layer."),
            Self::DifferentSizes => writeln!(f, "The two networks have different sizes."),
            Self::ZeroLayer => writeln!(f, "The network can't contain layers with 0 neurons.")
        }
    }
}

impl error::Error for NetworkError {}