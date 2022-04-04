use random_search::{feedforward::FeedForward1x1, activation_function::ActivationFunction, dataset::Dataset1x1, cost_function::CostFunction};

use std::ops::RangeInclusive;
use rand::{prelude::StdRng, SeedableRng};

fn main() {
    let mut mean_cost = 0.0;
    let experiments = 1usize;

    let mut stdrng = StdRng::from_entropy();

    let range = RangeInclusive::new(-10.0, 10.0);
    let dataset = Dataset1x1::<300>::new_random(&mut stdrng, f32::sin, range);

    for experiment in 1..=experiments {
        let mut ffnn = FeedForward1x1::new(
            &mut stdrng,
            vec![32; 3],
            ActivationFunction::ReLU,
            CostFunction::L2,
            dataset,
        ).unwrap();

        ffnn.random_search(&mut stdrng, 100_000, 16, false);

        println!("-------------------------");

        let final_loss = ffnn.evaluate_average_cost(true);

        mean_cost += final_loss;

        println!("Experiment {}: {}", experiment, final_loss);
    }
    
    println!("Average loss value: {}", mean_cost / (experiments as f32));
}