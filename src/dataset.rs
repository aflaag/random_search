use std::{ops::RangeInclusive, iter, slice};
use rand::Rng;

pub type Dataset1x1<const S: usize> = DatasetNxN<1, S>;

#[derive(Debug, Clone, Copy)]
pub struct DatasetNxN<const N: usize, const S: usize> {
    input: [[f32; N]; S],
    output: [[f32; N]; S],
}

impl<const N: usize, const S: usize> DatasetNxN<N, S> {
    pub fn new_random<R: Rng + ?Sized, F: Fn(f32) -> f32>(rng: &mut R, func: F, range: RangeInclusive<f32>) -> Self {
        let mut input = [(); S].map(|_| [(); N].map(|_| rng.gen_range(range.clone())));
        let mut output = [[0.0; N]; S];

        input.iter_mut().zip(output.iter_mut()).for_each(|(i, o)| {
            i.iter_mut().zip(o.iter_mut()).for_each(|(x, y)| *y = func(*x));
        });

        Self {
            input,
            output,
        }
    }

    pub fn iter_zip(&self) -> iter::Zip<slice::Iter<'_, [f32; N]>, slice::Iter<'_, [f32; N]>> {
        self.input.iter().zip(self.output.iter())
    }
}