use rand::{prelude::StdRng, SeedableRng};
use random_search::{feedforward::{FeedForward1x1, self}, activation_function::ActivationFunction, dataset::Dataset1x1};

fn main() {
    let mut stdrng = StdRng::from_entropy();

    let mut ffnn = FeedForward1x1::<300>::new(
        &mut stdrng,
        vec![32, 32, 32],
        ActivationFunction::ReLU,
        feedforward::STEP_SIZE,
    ).unwrap();

    ffnn.random_search(&mut stdrng, 25_000, 256);

    println!("-------------------------");

    let n = 5.0;
    let samples = 300;
    let m = (n * 2.0) / samples as f64;

    let mut cost = 0.0;

    for i in 0..samples {
        let x = m * i as f64 - n;

        let expected = x.sin();

        let output = ffnn.evaluate([x; 1])[(0, 0)];

        println!("({}, {})", x, output);

        cost += (output - expected) * (output - expected);
    }
    
    println!("{}", cost / (samples as f64));
}