# Supervised training script

This folder contains the Tensorflow script that is used to train the neural network models used to predict the next move, and the winner of a given board position. It expects SGF files as input, and will produce a Tensorflow model, which can be further transformed to a JSON model that _Dream Go_ can read.

## Running

```bash
make
./start_dev_container.sh
python -m dream_tf --start --config configs/15x192x6x128.json
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
python -m dream_tf --start --config configs/15x192x6x128-lz.json
python -m dream_tf --dump > dream_go.json
```

### Additional options

```
usage: dream_tf [-h] [--config CONFIG] [--model MODEL] [--name NAME] [--warm-start WARM_START] (--start | --resume | --verify | --dump)

Neural network optimizer for Dream Go.

optional arguments:
  -h, --help            show this help message and exit
  --start               start training of a new model
  --resume              resume training of a model
  --verify              evaluate the accuracy of a model
  --dump                print the weights of a model to standard output

optional configuration:
  --config CONFIG       the model configuration file
  --model MODEL         the directory that contains the model
  --name NAME           the name of this session
  --warm-start WARM_START
                        the model to warm-start from
```