use rand::{Rng, distributions::Standard, prelude::Distribution};

#[derive(Debug, Clone, Copy)]
pub struct DatasetNxN<T, const L: usize, const S: usize>
where
    T: Default + Copy,
    Standard: Distribution<T>
{
    input: [[T; L]; S],
    output: [[T; L]; S],
}

impl<T, const L: usize, const S: usize> DatasetNxN<T, L, S>
where
    T: Default + Copy,
    Standard: Distribution<T>
{
    pub fn new_random<F: Fn(T) -> T>(func: F) -> Self {
        let mut rng = rand::thread_rng();

        let mut input = [[T::default(); L]; S];
        let mut output = [[T::default(); L]; S];

        input.iter_mut().zip(output.iter_mut()).for_each(|(i, o)| {
            i.iter_mut().zip(o.iter_mut()).for_each(|(x, y)| {
                let random: T = rng.gen();

                *x = random;
                *y = func(random);
            });
        });

        Self {
            input,
            output,
        }
    }

    pub fn get_input(&self) -> [[T; L]; S] {
        self.input
    }

    pub fn get_output(&self) -> [[T; L]; S] {
        self.output
    }
}