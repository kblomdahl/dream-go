# CLOP - Confident Local Optimization for Noisy Black-Box Parameter Tuning

Environment for optimizing the parameters that affects the playing strength of Dream Go using CLOP, by Rémi Coulom. It optimizes the following parameters:

- `UCT_EXP`
- `VLOSS_CNT`
- `FPU_REDUCE`

## Running

Copy the most recent `dream_go.json` file to the current directory:

```bash
cargo build --release
bin/clop-console c < scripts/dream_go.clop
```
