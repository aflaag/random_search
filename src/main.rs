use rand::{prelude::StdRng, SeedableRng, thread_rng};
use random_search::{feedforward::FeedForward1x1, activation_function::ActivationFunction, dataset::Dataset1x1};

fn main() {
    let mut rng = thread_rng();

    let dataset = Dataset1x1::<f32, 3>::new_random(&mut rng, f32::sin);

    let mut stdrng = StdRng::from_entropy();

    let mut ffnn = FeedForward1x1::new(
        &mut stdrng,
        vec![3, 2],
        ActivationFunction::ReLU,
    ).unwrap();

    ffnn.load_dataset(dataset);

    ffnn.print_stuff();

    ffnn.random_search(&mut stdrng);
}