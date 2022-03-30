use nalgebra::{DimMax, DMatrix};
use rand::{prelude::StdRng, SeedableRng};
use random_search::{feedforward::{FeedForward1x1, self}, activation_function::ActivationFunction, dataset::Dataset1x1};

const S: usize = 300;

fn main() {
    let mut mean_cost = 0.0;
    let experiments = 5;

    for _ in 0..experiments {
        let mut stdrng = StdRng::from_entropy();

        let mut ffnn = FeedForward1x1::<{S}>::new(
            &mut stdrng,
            vec![32; 3],
            ActivationFunction::ReLU,
            feedforward::STEP_SIZE,
        ).unwrap();

        ffnn.random_search(&mut stdrng, 25_000, 32, true);

        println!("-------------------------");

        let n = 5.0;
        let m = (n * 2.0) / (S as f64);

        let mut cost = 0.0;

        for i in 0..S {
            let x = m * i as f64 - n;

            let expected = x.sin();

            let output = ffnn.evaluate(DMatrix::<f64>::from_iterator(1, 1, [x; 1]))[(0, 0)];

            println!("({}, {})", x, output);

            cost += (output - expected) * (output - expected);
        }

        mean_cost += cost / (S as f64);
    }
    
    println!("Average loss value: {}", mean_cost / (experiments as f64));
}