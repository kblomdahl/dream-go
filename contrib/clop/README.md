# CLOP - Confident Local Optimization for Noisy Black-Box Parameter Tuning

Environment for optimizing the parameters that affects the playing strength of Dream Go using CLOP, by Rémi Coulom. It optimizes the following parameters:

- `UCT_EXP`
- `VLOSS_CNT`
- `FPU_REDUCE`
- `SOFTMAX_TEMPERATURE`

## Running

Copy the most recent `dream_go.json` file to the current directory:

```bash
make
bin/clop-console c < scripts/all.clop
```

There are also some configuration files that allow you to tune one parameter at a time. The names should be self-explaining:

- `scripts/only_fpu_reduce.clop`
- `scripts/only_softmax_temperature.clop`
- `scripts/only_uct_exp.clop`
- `scripts/only_vloss_count.clop`

