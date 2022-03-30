# random_search

A neural network model that can approximate any non-linear function
by using the random search algorithm for the optimization of the loss function.

## Experiments

| experiments | target_function | S | networks | epochs | layers_sizes | STEP_SIZE | raw mean loss | mean loss % |
|-------------|------|---|----------|--------|--------------|-----------|----------|-------------|
| 5 | sin(x) | 300 | 16 | 25_000 | 32; 3 | 1e-4 | 0.0022145095302255496 | 0.221% |
| 5 | sin(x) | 300 | 16 | 25_000 | 8; 10 | 1e-4 | 0.5271807633104308 | 52.718% |
| 5 | sin(x) | 300 | 16 | 25_000 | 16; 5 | 1e-5 | 0.5271807632693248 | 52.718% |
| 5 | sin(x) | 300 | 16 | 25_000 | 16; 4 | 1e-4 | 0.06721093900301409 | 6.721% |
| 5 | sin(x) | 300 | 16 | 25_000 | 16; 3 | 1e-5 | 0.52699025347444 | 52.699% |
| 5 | sin(x) | 300 | 128 | 25_000 | 32; 3 | 1e-4 | 0.00012897642800016362 | 0.013% |
| 5 | sin(x) | 300 | 32 | 25_000 | 32; 4 | 1e-4 | 0.0009437770169172481 | 0.094% |
| 5 | x^3 - 2x^2 - 6x | 300 | 32 | 25_000 | 32; 4 | 1e-4 | 1390.5025496471226 | 139050.255% |
| 5 | sin(x) | 300 | 128 | 25_000 | 8; 6 | 1e-7 | 0.5271807632690662 | 52.718% |
| 5 | cos(x) | 300 | 32 | 25_000 | 32; 3 | 1e-4 | 0.006531106146449706 | 0.653% |
| 5 | sin(x) | 300 | 32 | 25_000 | 32; 2 | 1e-4 | 0.0007336470162722 | 0.073% |

## TODO list

- [x] implement dataset struct
- [x] implement activation function enum
- [x] change the struct name with the NxN thing
- [x] change the struct accordingly
- [x] implement the average
- [x] implement ffnn error enum
- [x] implement addassign for the ffnn
- [x] check if the summations between ffnns is doable, somehow
- [x] change rng in dataset
- [x] change new for ffnn with normal distribution
- [x] implement the random search algorithm
- [x] fix the standard gaussian error
- [x] understand what the fuck is going on
- [x] fix the problem with rayon with the rng
- [x] implement rayon, super important
- [x] implement experiment system
- [x] change generate_matrix_from_iterator()
- [x] use windows in ffnn::new()
- [x] implement seeds for the best ffnn
- [x] make evaluate() to take same input and output
- [ ] make a decent cost function, it sucks
- [ ] go back to f32 for a little bit of performance improvements
- [ ] consider using the dataset thing still
- [ ] make a better add method
- [ ] implement nn trait
- [ ] implement dataset trait
- [ ] implement dataset error enum (useful?)
- [ ] 0 sided matrices and vectors are allowed (?)
- [ ] document everything
- [ ] consider implementing the genetic algorithm
- [ ] consider parallelizing evaluate()
