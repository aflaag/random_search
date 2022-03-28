use rand::{prelude::StdRng, SeedableRng, thread_rng};
use random_search::{feedforward::{FeedForward1x1, self}, activation_function::ActivationFunction, dataset::Dataset1x1};

fn main() {
    let mut rng = thread_rng();
    let mut stdrng = StdRng::from_entropy();

    let mut ffnn = FeedForward1x1::new(
        &mut stdrng,
        vec![16, 16, 16],
        ActivationFunction::ReLU,
        feedforward::STEP_SIZE,
    ).unwrap();

    let n = 5.0;

    let dataset = Dataset1x1::<f32, 128>::new_random(
        &mut rng,
        f32::sin,
        -n..=n
    );

    ffnn.load_dataset(dataset);

    let samples = 64;
    let m = (n * 2.0) / samples as f32;

    for x in 0..samples {
        let y = m * x as f32 - n;

        println!("({}, {})", y, ffnn.evaluate([y])[(0, 0)]); // index 0 is safe
    }

    ffnn.random_search(&mut stdrng, 1_000, 256);

    println!("-------------------------");

    let samples = 64;
    let m = (n * 2.0) / samples as f32;

    for x in 0..samples {
        let y = m * x as f32 - n;

        println!("({}, {})", y, ffnn.evaluate([y])[(0, 0)]); // index 0 is safe
    }
}