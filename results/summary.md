# Results

Tables written by `romlab report` from the JSON files.

## Reduction study

128 training, 64 validation and 256 test fields per family, seeds [0, 1, 2], grid of 256 points. Test error is the mean relative L2 error, averaged over seeds.

### pulse

Gaussian bump (width 0.04) advected on [0, 1]; parameter: centre in [0.2, 0.8]

| n | POD | quadratic | polynomial | POD + MLP | autoencoder |
|---|---|---|---|---|---|
| 1 | 0.76 | 0.67 | 0.51 | 0.5 | 0.048 |
| 2 | 0.63 | 0.36 | 1.3e-05 | 0.007 | 0.0031 |
| 3 | 0.51 | 0.1 | 2.9e-06 | 0.0054 | 0.004 |
| 4 | 0.4 | 0.015 | 1.6e-06 | 0.0032 | 0.0026 |
| 6 | 0.23 | 2.5e-04 | 1.7e-06 | 0.0015 | 0.0025 |
| 8 | 0.12 | 6.3e-05 | 1.5e-06 | 5.8e-04 | 0.0026 |
| 12 | 0.023 | 5.2e-06 | 2.0e-07 | 1.2e-04 | 0.0025 |
| 16 | 0.0031 | 3.0e-07 | 2.7e-08 | 4.0e-05 | 0.0024 |

### front

tanh front (thickness 0.01) on [0, 1]; parameter: position in [0.2, 0.8]

| n | POD | quadratic | polynomial | POD + MLP | autoencoder |
|---|---|---|---|---|---|
| 1 | 0.28 | 0.21 | 0.059 | 0.05 | 0.021 |
| 2 | 0.21 | 0.13 | 0.0052 | 0.026 | 0.011 |
| 3 | 0.17 | 0.086 | 0.0013 | 0.017 | 0.0097 |
| 4 | 0.14 | 0.057 | 6.2e-04 | 0.012 | 0.0093 |
| 6 | 0.11 | 0.026 | 7.1e-04 | 0.0076 | 0.0089 |
| 8 | 0.087 | 0.013 | 7.3e-04 | 0.005 | 0.009 |
| 12 | 0.06 | 0.0039 | 4.7e-04 | 0.0033 | 0.0094 |
| 16 | 0.042 | 0.0016 | 4.7e-04 | 0.0022 | 0.009 |

### soliton

KdV one-soliton on [0, 20]; parameters: speed in [1, 4], position in [6, 14]

| n | POD | quadratic | polynomial | POD + MLP | autoencoder |
|---|---|---|---|---|---|
| 1 | 0.56 | 0.5 | 0.46 | 0.46 | 0.27 |
| 2 | 0.39 | 0.18 | 0.0069 | 0.011 | 0.0081 |
| 3 | 0.28 | 0.071 | 4.2e-04 | 0.0081 | 0.0075 |
| 4 | 0.21 | 0.029 | 2.7e-04 | 0.0049 | 0.0068 |
| 6 | 0.1 | 0.0057 | 7.4e-05 | 0.0021 | 0.0067 |
| 8 | 0.047 | 0.0012 | 6.6e-05 | 0.0011 | 0.0067 |
| 12 | 0.011 | 8.2e-05 | 2.5e-05 | 3.5e-04 | 0.0066 |
| 16 | 0.0027 | 1.6e-05 | 1.6e-05 | 1.8e-04 | 0.0067 |

### burgers

Cole-Hopf solution of viscous Burgers (nu = 0.02) on the periodic unit interval; parameters: eps in [0.90, 0.995], shift in [-0.25, 0.25]

| n | POD | quadratic | polynomial | POD + MLP | autoencoder |
|---|---|---|---|---|---|
| 1 | 0.68 | 0.6 | 0.51 | 0.5 | 0.28 |
| 2 | 0.53 | 0.34 | 0.032 | 0.062 | 0.052 |
| 3 | 0.42 | 0.21 | 0.015 | 0.049 | 0.034 |
| 4 | 0.33 | 0.14 | 0.0092 | 0.039 | 0.031 |
| 6 | 0.22 | 0.067 | 0.008 | 0.025 | 0.03 |
| 8 | 0.16 | 0.029 | 0.0074 | 0.021 | 0.029 |
| 12 | 0.1 | 0.0093 | 0.0044 | 0.012 | 0.029 |
| 16 | 0.06 | 0.0047 | 0.0028 | 0.0078 | 0.029 |

### Smallest latent dimension with mean test error below 1%

| family | POD | quadratic | polynomial | POD + MLP | autoencoder |
|---|---|---|---|---|---|
| pulse | 16 | 6 | 2 | 2 | 2 |
| front | >16 | 12 | 2 | 6 | 3 |
| soliton | 16 | 6 | 2 | 3 | 2 |
| burgers | >16 | 12 | 4 | 16 | >16 |

### Decay of the POD error

Slope of log error against log n, n from 2.

| family | slope | error at n=2 | error at n=16 |
|---|---|---|---|
| pulse | -2.39 | 0.63 | 0.0031 |
| front | -0.76 | 0.21 | 0.042 |
| soliton | -2.34 | 0.39 | 0.0027 |
| burgers | -1.04 | 0.53 | 0.06 |

### Decoder sensitivity at n = 4

Noise: change of the decoded field relative to its norm when each latent code gets noise of 5% of its spread (median and 95th percentile over test fields). Jacobian: 95th percentile of the spectral norm of the decoder Jacobian in unit-spread latent coordinates, divided by the mean field norm.

| family | decoder | test error | noise median | noise p95 | Jacobian p95 |
|---|---|---|---|---|---|
| pulse | POD | 0.4 | 0.039 | 0.068 | 0.44 |
| pulse | quadratic | 0.015 | 1.9 | 6.1 | 85 |
| pulse | polynomial | 1.6e-06 | 0.054 | 0.14 | 2.5 |
| pulse | POD + MLP | 0.0032 | 0.063 | 0.13 | 1.5 |
| pulse | autoencoder | 0.0026 | 0.041 | 0.083 | 0.88 |
| front | POD | 0.14 | 0.018 | 0.042 | 0.35 |
| front | quadratic | 0.057 | 21 | 1.3e+02 | 1.6e+03 |
| front | polynomial | 6.2e-04 | 3.2 | 38 | 9e+02 |
| front | POD + MLP | 0.012 | 0.03 | 0.064 | 0.82 |
| front | autoencoder | 0.0093 | 0.027 | 0.065 | 0.86 |
| soliton | POD | 0.21 | 0.036 | 0.071 | 0.54 |
| soliton | quadratic | 0.029 | 0.061 | 0.12 | 1.5 |
| soliton | polynomial | 2.7e-04 | 0.13 | 0.4 | 5.3 |
| soliton | POD + MLP | 0.0049 | 0.043 | 0.097 | 0.81 |
| soliton | autoencoder | 0.0068 | 0.035 | 0.083 | 0.74 |
| burgers | POD | 0.33 | 0.041 | 0.077 | 0.55 |
| burgers | quadratic | 0.14 | 0.13 | 0.3 | 3.4 |
| burgers | polynomial | 0.0092 | 0.46 | 1.9 | 30 |
| burgers | POD + MLP | 0.039 | 0.065 | 0.13 | 1.7 |
| burgers | autoencoder | 0.031 | 0.044 | 0.096 | 0.84 |

## Operator study, 4-dimensional inputs, 4000 latent epochs

Burgers, nu = 0.02, T = 0.4, 256 grid points, seeds [0, 1, 2], latent sizes [2, 4, 8] (chosen per fit on validation fields). Test error: mean over seeds with the range over seeds in brackets.

| training fields | FNO | fixed POD decoder | trained linear decoder | linear + MLP decoder |
|---|---|---|---|---|
| 32 | 0.044 (0.04 to 0.05) | 0.42 (0.4 to 0.44) | 0.44 (0.43 to 0.44) | 0.27 (0.23 to 0.29) |
| 64 | 0.018 (0.016 to 0.021) | 0.24 (0.23 to 0.24) | 0.24 (0.23 to 0.25) | 0.12 (0.12 to 0.12) |
| 128 | 0.0074 (0.0068 to 0.0077) | 0.13 (0.13 to 0.13) | 0.14 (0.13 to 0.14) | 0.043 (0.043 to 0.043) |
| 256 | 0.0044 (0.0039 to 0.0047) | 0.1 (0.1 to 0.1) | 0.1 (0.099 to 0.1) | 0.022 (0.02 to 0.022) |
| 512 | 0.0036 (0.0031 to 0.0041) | 0.088 (0.087 to 0.088) | 0.087 (0.087 to 0.088) | 0.012 (0.011 to 0.013) |
| 1024 | 0.0031 (0.0029 to 0.0032) | 0.084 (0.084 to 0.085) | 0.085 (0.084 to 0.085) | 0.009 (0.009 to 0.0091) |

Trainable parameters.

| model | latent size | parameters |
|---|---|---|
| FNO | none | 139,745 |
| fixed POD decoder | 2 | 17,410 |
| fixed POD decoder | 4 | 17,668 |
| fixed POD decoder | 8 | 18,184 |
| trained linear decoder | 2 | 18,178 |
| trained linear decoder | 4 | 18,948 |
| trained linear decoder | 8 | 20,488 |
| linear + MLP decoder | 2 | 117,378 |
| linear + MLP decoder | 4 | 118,404 |
| linear + MLP decoder | 8 | 120,456 |

Paired differences of test error (first minus second, negative means the first is better), mean with a 95% bootstrap interval over test fields pooled over seeds.

| training fields | FNO minus fixed POD decoder | linear + MLP decoder minus FNO | linear + MLP decoder minus trained linear decoder | linear + MLP decoder minus fixed POD decoder |
|---|---|---|---|---|
| 32 | -0.38 [-0.39, -0.37] | 0.22 [0.21, 0.24] | -0.17 [-0.18, -0.16] | -0.16 [-0.17, -0.15] |
| 64 | -0.22 [-0.23, -0.21] | 0.1 [0.093, 0.11] | -0.12 [-0.12, -0.11] | -0.12 [-0.12, -0.11] |
| 128 | -0.13 [-0.13, -0.12] | 0.036 [0.034, 0.038] | -0.092 [-0.096, -0.089] | -0.092 [-0.095, -0.088] |
| 256 | -0.097 [-0.1, -0.094] | 0.017 [0.016, 0.019] | -0.08 [-0.082, -0.077] | -0.08 [-0.083, -0.078] |
| 512 | -0.084 [-0.087, -0.081] | 0.0084 [0.0077, 0.0092] | -0.075 [-0.078, -0.073] | -0.076 [-0.078, -0.073] |
| 1024 | -0.081 [-0.084, -0.078] | 0.006 [0.0055, 0.0064] | -0.075 [-0.078, -0.073] | -0.075 [-0.078, -0.073] |

## Operator study, 4-dimensional inputs, 20000 latent epochs

Burgers, nu = 0.02, T = 0.4, 256 grid points, seeds [0], latent sizes [4, 8] (chosen per fit on validation fields). Test error: mean over seeds with the range over seeds in brackets.

| training fields | fixed POD decoder | linear + MLP decoder |
|---|---|---|
| 128 | 0.13 (0.13 to 0.13) | 0.045 (0.045 to 0.045) |
| 1024 | 0.08 (0.08 to 0.08) | 0.0039 (0.0039 to 0.0039) |

Trainable parameters.

| model | latent size | parameters |
|---|---|---|
| fixed POD decoder | 4 | 17,668 |
| fixed POD decoder | 8 | 18,184 |
| linear + MLP decoder | 4 | 118,404 |
| linear + MLP decoder | 8 | 120,456 |

Paired differences of test error (first minus second, negative means the first is better), mean with a 95% bootstrap interval over test fields pooled over seeds.

| training fields | linear + MLP decoder minus fixed POD decoder |
|---|---|
| 128 | -0.09 [-0.095, -0.084] |
| 1024 | -0.076 [-0.081, -0.071] |

## Operator study, 12-dimensional inputs, 4000 latent epochs

Burgers, nu = 0.02, T = 0.4, 256 grid points, seeds [0, 1, 2], latent sizes [6, 12, 24] (chosen per fit on validation fields). Test error: mean over seeds with the range over seeds in brackets.

| training fields | FNO | fixed POD decoder | trained linear decoder | linear + MLP decoder |
|---|---|---|---|---|
| 64 | 0.062 (0.056 to 0.066) | 0.58 (0.58 to 0.59) | 0.58 (0.57 to 0.59) | 0.56 (0.55 to 0.57) |
| 256 | 0.016 (0.014 to 0.017) | 0.37 (0.36 to 0.38) | 0.38 (0.38 to 0.4) | 0.17 (0.16 to 0.18) |
| 1024 | 0.0068 (0.0067 to 0.0068) | 0.1 (0.097 to 0.1) | 0.1 (0.097 to 0.11) | 0.043 (0.04 to 0.045) |

Trainable parameters.

| model | latent size | parameters |
|---|---|---|
| FNO | none | 139,745 |
| fixed POD decoder | 6 | 18,950 |
| fixed POD decoder | 12 | 19,724 |
| fixed POD decoder | 24 | 21,272 |
| trained linear decoder | 6 | 20,742 |
| trained linear decoder | 12 | 23,052 |
| trained linear decoder | 24 | 27,672 |
| linear + MLP decoder | 6 | 120,454 |
| linear + MLP decoder | 12 | 123,532 |
| linear + MLP decoder | 24 | 129,688 |

Paired differences of test error (first minus second, negative means the first is better), mean with a 95% bootstrap interval over test fields pooled over seeds.

| training fields | FNO minus fixed POD decoder | linear + MLP decoder minus FNO | linear + MLP decoder minus trained linear decoder | linear + MLP decoder minus fixed POD decoder |
|---|---|---|---|---|
| 64 | -0.52 [-0.54, -0.51] | 0.5 [0.48, 0.52] | -0.019 [-0.025, -0.012] | -0.024 [-0.031, -0.016] |
| 256 | -0.36 [-0.37, -0.35] | 0.15 [0.15, 0.16] | -0.22 [-0.22, -0.21] | -0.21 [-0.21, -0.2] |
| 1024 | -0.093 [-0.096, -0.089] | 0.037 [0.035, 0.038] | -0.061 [-0.064, -0.059] | -0.056 [-0.059, -0.054] |

## Operator study, 12-dimensional inputs, 20000 latent epochs

Burgers, nu = 0.02, T = 0.4, 256 grid points, seeds [0], latent sizes [12, 24] (chosen per fit on validation fields). Test error: mean over seeds with the range over seeds in brackets.

| training fields | fixed POD decoder | linear + MLP decoder |
|---|---|---|
| 1024 | 0.1 (0.1 to 0.1) | 0.035 (0.035 to 0.035) |

Trainable parameters.

| model | latent size | parameters |
|---|---|---|
| fixed POD decoder | 12 | 19,724 |
| fixed POD decoder | 24 | 21,272 |
| linear + MLP decoder | 12 | 123,532 |
| linear + MLP decoder | 24 | 129,688 |

Paired differences of test error (first minus second, negative means the first is better), mean with a 95% bootstrap interval over test fields pooled over seeds.

| training fields | linear + MLP decoder minus fixed POD decoder |
|---|---|
| 1024 | -0.068 [-0.073, -0.063] |

## Output reconstruction, 4-dimensional inputs

Reducers fitted on 1024 training outputs of the Burgers task and scored on the same test outputs as the operator study, with each reducer's own encoding of the true output as the latent code. It shows how much the bottleneck alone costs for each decoder family; it is not a strict bound, since the operator models train their decoders end to end. Mean over seeds [0].

| latent size | POD | POD + MLP | autoencoder |
|---|---|---|---|
| 2 | 0.44 | 0.19 | 0.16 |
| 4 | 0.23 | 0.015 | 0.011 |
| 8 | 0.078 | 0.0028 | 0.0092 |

## Output reconstruction, 12-dimensional inputs

Reducers fitted on 1024 training outputs of the Burgers task and scored on the same test outputs as the operator study, with each reducer's own encoding of the true output as the latent code. It shows how much the bottleneck alone costs for each decoder family; it is not a strict bound, since the operator models train their decoders end to end. Mean over seeds [0].

| latent size | POD | POD + MLP | autoencoder |
|---|---|---|---|
| 6 | 0.14 | 0.0089 | 0.014 |
| 12 | 0.031 | 0.0012 | 0.01 |
| 24 | 0.0022 | 7.8e-05 | 0.011 |
