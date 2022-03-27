use random_search::{ffnn::FeedForwardNxN, activation_function::ActivationFunction, dataset::DatasetNxN};

fn main() {
    let dataset = DatasetNxN::<f32, 1, 3>::new_random(f32::sin);

    let mut ffnn = FeedForwardNxN::new(
        vec![3, 2],
        ActivationFunction::ReLU,
        dataset,
    );

    ffnn.print_stuff();
    ffnn.train();
}