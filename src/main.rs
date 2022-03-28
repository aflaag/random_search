use rand::{prelude::StdRng, SeedableRng, thread_rng};
use random_search::{feedforward::FeedForward1x1, activation_function::ActivationFunction, dataset::Dataset1x1};

fn main() {
    let mut rng = thread_rng();

    let mut ffnn = FeedForward1x1::new(
        &mut rng,
        vec![3, 2],
        ActivationFunction::ReLU,
        1.0,
    ).unwrap();

    let n = 5.0;

    let dataset = Dataset1x1::<f32, 1000>::new_random(
        &mut rng,
        f32::sin,
        -n..=n
    );

    ffnn.load_dataset(dataset);

    let mut stdrng = StdRng::from_entropy();

    ffnn.random_search(&mut stdrng, 100);

    let samples = 64;
    let m = (n * 2.0) / samples as f32;

    for x in 0..samples {
        let y = m * x as f32 - n;

        println!("({}, {})", y, ffnn.evaluate([y])[(0, 0)]); // index 0 is safe
    }
}