# Contributing

Thank you for your interest in irsim.

## What is welcome right now

- **Bug reports** — something crashes, gives wrong numbers, or does not match the documentation.
- **Physics questions and corrections** — an equation, constant or material property you think is
  wrong. Please cite a source.
- **Feature requests** — a camera, band, material, scene or output you need.
- **Validation data** — pointers to public measured data (spectral responses, emissivity tables,
  real infrared imagery) the simulator could be checked against.

Open any of these from the
[issue page](https://github.com/reza-shahriari/isaac-thermal-camera-simulation/issues/new/choose),
which offers a form for each.

## Pull requests are not accepted yet

The project is published under a use-only licence (see [LICENSE](LICENSE), sections 2 and 4),
which does not allow modified versions, so pull requests cannot be merged at the moment. Please
describe the change you have in mind in an issue instead; anything suggested there may be used in
the project. This is expected to change in a later release.

## Writing a useful bug report

- **Commit hash** (`git rev-parse --short HEAD`), Python version, and whether Isaac Sim was involved.
- **The command you ran** and the full error or output.
- **The config files** involved (sensor, scene, material), or the smallest one that shows the
  problem.
- **For a physics bug:** the value you got, the value you expected, and where the expected value
  comes from. [`docs/physics-model.md`](docs/physics-model.md) is the specification the code
  follows; quoting the section number helps a lot.

## Running the tests locally

The physics core needs only Python 3.10+ and NumPy; no GPU or Isaac Sim:

```bash
make install   # editable install + dev dependencies
make luts      # generate the band lookup tables
make test      # unit + golden tests (fast tier, all cores)
make check     # lint, types and the tests your change affects -- before every commit
make check-full  # the same with every test -- before a push; CI runs this
```

Integration tests need Isaac Sim and are skipped by default. See
[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full command list and project layout.

## Conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report a
security problem privately as described in [SECURITY.md](SECURITY.md), not in a public issue.
