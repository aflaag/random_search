use rand::{Rng, distributions::{Standard, uniform::{SampleUniform, SampleRange}}, prelude::Distribution};

pub type Dataset1x1<T, const S: usize> = DatasetNxN<T, 1, S>;

#[derive(Debug, Clone, Copy)]
pub struct DatasetNxN<T, const N: usize, const S: usize>
where
    T: Default + Copy,
    Standard: Distribution<T>
{
    input: [[T; N]; S],
    output: [[T; N]; S],
}

impl<T, const N: usize, const S: usize> DatasetNxN<T, N, S>
where
    T: Default + Copy + SampleUniform,
    Standard: Distribution<T>
{
    pub fn new_random<R, F, U>(rng: &mut R, func: F, range: U) -> Self
    where
        R: Rng + ?Sized,
        F: Fn(T) -> T,
        U: SampleRange<T> + Clone,
    {
        let mut input = [[T::default(); N]; S];
        let mut output = [[T::default(); N]; S];

        input.iter_mut().zip(output.iter_mut()).for_each(|(i, o)| {
            i.iter_mut().zip(o.iter_mut()).for_each(|(x, y)| {
                let random: T = rng.gen_range(range.clone());

                *x = random;
                *y = func(random);
            });
        });

        Self {
            input,
            output,
        }
    }

    pub fn get_input(&self) -> [[T; N]; S] {
        self.input
    }

    pub fn get_output(&self) -> [[T; N]; S] {
        self.output
    }
}

impl<T, const N: usize, const S: usize> Default for DatasetNxN<T, N, S>
where
    T: Default + Copy,
    Standard: Distribution<T>
{
    fn default() -> Self {
        Self {
            input: [[T::default(); N]; S],
            output: [[T::default(); N]; S],
        }
    }
}