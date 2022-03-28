use rand::{prelude::StdRng, SeedableRng};
use random_search::{feedforward::FeedForward1x1, activation_function::ActivationFunction, dataset::Dataset1x1};

fn main() {
    let dataset = Dataset1x1::<f32, 3>::new_random(f32::sin);

    let mut ffnn = FeedForward1x1::new(
        vec![3, 2],
        ActivationFunction::ReLU,
        dataset,
    ).unwrap();

    ffnn.print_stuff();

    let mut stdrng = StdRng::from_entropy();

    ffnn.random_search(&mut stdrng);
}