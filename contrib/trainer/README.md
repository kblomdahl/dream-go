# Supervised training script

This folder contains the Tensorflow script that is used to train the neural network models used to predict the next move, and the winner of a given board position. It expects SGF files as input, and will produce a Tensorflow model, which can be further transformed to a JSON model that _Dream Go_ can read.

## Running

```bash
make
./start_dev_container.sh
python -m dream_tf --start [files]
python -m dream_tf --dump > dream_go.json
```
