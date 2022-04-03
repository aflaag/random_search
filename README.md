# random_search

A neural network model that can approximate any non-linear function
by using the random search algorithm for the optimization of the loss function.

## Important

If you are coming from the [article on Medium](https://medium.com/@alessio.bandiera02/but-what-is-artificial-intelligence-exactly-c1acbe3c3cd1), keep in mind that the code is still partially incomplete and, most importantly, not fully documented, but you can figure things out.

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
