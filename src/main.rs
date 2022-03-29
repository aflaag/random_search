use rand::{prelude::StdRng, SeedableRng, thread_rng};
use random_search::{feedforward::{FeedForward1x1, self}, activation_function::ActivationFunction, dataset::Dataset1x1};

fn main() {
    // let mut rng = thread_rng();
    let mut stdrng = StdRng::from_entropy();

    let mut ffnn = FeedForward1x1::<300>::new(
        &mut stdrng,
        // vec![16, 16, 16],
        vec![32, 32, 32],
        ActivationFunction::ReLU,
        feedforward::STEP_SIZE,
        // 0.5,
    ).unwrap();

    // let n = 5.0;

    // let dataset = Dataset1x1::<f64, 200>::new_random(
    //     &mut rng,
    //     f64::sin,
    //     -n..=n
    // );

    // ffnn.load_dataset(dataset);

    // let samples = 64;
    // let m = (n * 2.0) / samples as f64;

    // for x in 0..samples {
    //     let y = m * x as f64 - n;

    //     println!("({}, {})", y, ffnn.evaluate([y])[(0, 0)]); // index [(0, 0)] is safe
    // }

    ffnn.random_search(&mut stdrng, 25_000, 512);

    println!("-------------------------");

    // let samples = 64;
    // let m = (n * 2.0) / samples as f64;

    // for x in 0..samples {
    //     let y = m * x as f64 - n;

    //     println!("({}, {})", y, ffnn.evaluate([y])[(0, 0)]); // index [(0, 0)] is safe
    // }

    // let mut cost = 0.0;
    // let mut a = 0.0;

    // for (x_vec, y_vec) in dataset.get_input().iter().zip(dataset.get_output()) {
    //     let x = x_vec[0];

    //     let output = ffnn.evaluate([x])[(0, 0)];

    //     let expected = x.sin();
    //     let y = y_vec[0];

    //     a += (y - expected) * (y - expected);

    //     cost += (output - expected) * (output - expected);

    //     println!("({}, {})", x, output);
    // }

    // println!("{} {}", cost / 200.0, a / 200.0);

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