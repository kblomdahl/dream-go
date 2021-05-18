# Supervised training script

This folder contains the Tensorflow script that is used to train the neural network models used to predict the next move, and the winner of a given board position. It expects SGF files as input, and will produce a Tensorflow model, which can be further transformed to a JSON model that _Dream Go_ can read.

## Running

```bash
make
./start_dev_container.sh
python -m dream_tf --start [files]
python -m dream_tf --dump > dream_go.json
```

### Tensorboard

The training script will dump a lot of metrics to the `models/` directory that can be read by tensorboard.

```bash
./start_dev_container.sh
tensorboard --logdir models/
```

### Semi-supervised learning with LZ

If you wish to bootstrap the weights from leela-zero you can do so by [downloading some weights](https://zero.sjeng.org/). You will still need to provide the engine with board positions to learn from, but the policy and value from the SGF file will be replaced by the prediction from the leela-zero model.

```bash
make
curl https://zero.sjeng.org/networks/0e9ea880fd3c4444695e8ff4b8a36310d2c03f7c858cadd37af3b76df1d1d15f.gz > models/0e9ea880.gz
./start_dev_container.sh
python -m dream_tf --start --lz-weights models/0e9ea880.gz [files]
python -m dream_tf --dump > dream_go.json
```

### Additional options

```
usage: dream_tf [-h] [--batch-size N] [--test-batches N] [--warm-start M]
                [--steps N] [--model MODEL] [--name NAME]
                [--lz-weights LZ_WEIGHTS] [--debug] [--deterministic]
                [--profile] [--num-channels N] [--num-blocks N]
                [--num-samples N] [--mask M]
                (--start | --resume | --verify | --dump | --print)
                ...

Neural network optimizer for Dream Go.

positional arguments:
  files                 The binary features files.

optional arguments:
  -h, --help            show this help message and exit
  --start               start training of a new model
  --resume              resume training of an existing model
  --verify              evaluate the accuracy of a model
  --dump                print the weights of a model to standard output
  --print               print the value of the given tensor

optional configuration:
  --batch-size N        the number of examples per mini-batch
  --test-batches N      the number of mini-batches to reserve for evaluation
  --warm-start M        initialize weights from the given model
  --steps N             the total number of examples to train over
  --model MODEL         the directory that contains the model
  --name NAME           the name of this session
  --lz-weights LZ_WEIGHTS
                        leela-zero weights to use for semi-supervised learning
  --debug               enable command-line debugging
  --deterministic       enable deterministic mode
  --profile             enable profiling

model configuration:
  --num-channels N      the number of channels per residual block
  --num-blocks N        the number of residual blocks
  --num-samples N       the number of global average pooling samples
  --mask M              mask to multiply features with
```