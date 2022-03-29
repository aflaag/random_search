extern crate nalgebra as na;

use crate::{activation_function::ActivationFunction, dataset::DatasetNxN};

use std::{error, fmt, ops};
use na::DMatrix;
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

pub const STEP_SIZE: f64 = 0.0001;

pub type FeedForward1x1<const S: usize> = FeedForwardNxN<1, S>;

#[derive(Debug, Clone)]
pub struct FeedForwardNxN<const N: usize, const S: usize> {
    layers_sizes: Vec<usize>,
    layers_number: usize,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
    activation_function: ActivationFunction,
    dataset: DatasetNxN<f64, N, S>,
}

impl<const N: usize, const S: usize> FeedForwardNxN<N, S> {
    fn generate_matrix_from_iterator<R: Rng + ?Sized>(m: usize, n: usize, step_size: f64, rng: &mut R) -> DMatrix<f64> {
        DMatrix::from_iterator(m, n,
            (0..m * n)
                .zip(rng.sample_iter(StandardNormal))
                .map(|(_, random): (usize, f64)| random * step_size)
        )
    }

    pub fn new<R: Rng + ?Sized>(rng: &mut R, layers_sizes: Vec<usize>, activation_function: ActivationFunction, step_size: f64) -> Result<Self, NetworkError> {
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
        let mut biases: Vec<DMatrix<f64>> = layers_sizes.iter().map(|n|
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

    pub fn load_dataset(&mut self, dataset: DatasetNxN<f64, N, S>) {
        self.dataset = dataset;
    }

    fn evaluate_average_cost(&self) -> f64 {
        // self.dataset
        //     .get_input()
        //     .iter()
        //     .zip(self.dataset.get_output())
        //     .map(|(input, output)| {
        //         output
        //             .iter()
        //             .zip(self.evaluate(*input).iter())
        //             .map(|(o, y)| (o - y) * (o - y))
        //             .sum::<f64>()
        //     })
        //     .sum::<f64>() / (S as f64)

        // let mut cost = 0.0;

        // for x_vec in self.dataset.get_input()

        // let mut t = 0.0;
        // // let mut rng = thread_rng();
        // let mut rng = StdRng::from_entropy();

        // for _ in 0..128 {
        //     let x: f64 = rng.sample::<f64, _>(StandardNormal);
        //     // let x: f64 = 10.0 / (1.0 + rng.sample::<f64, _>(StandardNormal)) - 5.0;
        //     // println!("{}", x);

        //     t += self.evaluate([x as f64; N])
        //         .iter()
        //         .map(|output| {
        //             let expected = (x as f64).sin();

        //             (output - expected) * (output - expected)
        //         })
        //         .sum::<f64>();
        // }

        // t / (S as f64)

        let n = 5.0;
        let m = (n * 2.0) / (S as f64);

        let mut cost = 0.0;

        for i in 0..S {
            let x = m * i as f64 - n;

            let expected = x.sin();

            let output = self.evaluate([x; N])[(0, 0)];

            cost += (output - expected) * (output - expected);
        }

        cost / (S as f64)
    }

    pub fn random_search<R: Rng + ?Sized + Sync + Send + Clone>(&mut self, rng: &mut R, epochs: usize, networks: usize) {
        for epoch in 0..epochs {
            *self += (0..networks)
                // .into_par_iter()
                // .map_with(rng.clone(), |rng, _| {
                .map(|_| {
                    FeedForwardNxN::<N, S>::new(
                        rng,
                        // &mut rng.clone(),
                        self.layers_sizes.clone(),
                        self.activation_function,
                        STEP_SIZE,
                    ).unwrap()
                }).min_by(|x, y|
                    (x.clone() + self.clone()).unwrap()
                    // x
                        .evaluate_average_cost()
                        // .partial_cmp(&y.evaluate_average_cost())
                        .partial_cmp(&(y.clone() + self.clone()).unwrap().evaluate_average_cost())
                        .unwrap()
                ).unwrap();

            println!("({}, {})", epoch, self.evaluate_average_cost());
        }
    }

    pub fn evaluate(&self, input: [f64; N]) -> DMatrix<f64> {
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
        
        // (0..self.layers_number).into_par_iter().for_each(|i| {
        //     x_vec = &self.weights[i] * x_vec + &self.biases[i];

        //     x_vec.iter_mut().for_each(|x| *x = (self.activation_function.function())(*x));
        // });

        // in fact, here we index the weights
        // and the biases at layers_number exactly,
        // because the length of those vectors are
        // layers_number + 1, since layer_number
        // is just the number of hidden layers
        &self.weights[self.layers_number] * x_vec + &self.biases[self.layers_number]
    }

    // TODO: debug
    pub fn print_stuff(&self) {
    }
}

// TODO: remove this, terrible
impl<const N: usize, const S: usize> ops::Add for FeedForwardNxN<N, S> {
    type Output = Result<Self, NetworkError>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.layers_sizes != rhs.layers_sizes {
            return Err(NetworkError::DifferentSizes)
        } else {
            let mut result = self.clone();

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
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyLayers => writeln!(f, "The network must have at least 1 hidden layer."),
            Self::DifferentSizes => writeln!(f, "The two networks have different sizes."),
        }
    }
}

impl error::Error for NetworkError {}