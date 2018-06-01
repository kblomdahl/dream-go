# Dream Go - All day, every day

Dream Go is an independent implementation of the algorithms and concepts presented by DeepMind in their [Master the Game of Go without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) paper with a few modifications to (maybe) make it feasible to develop a strong player without access to a supercomputer on the scale of [Sunway TaihuLight](https://en.wikipedia.org/wiki/Sunway_TaihuLight).

* Human games are used to bootstrap the network weights.
* Additional (synthetic) features inspired by [AlphaGo](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) and [DeepForest](https://arxiv.org/pdf/1511.06410.pdf) are used during training and inference.
* A self learning procedure inspired by [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/pdf/1705.08439.pdf) is used.

## Dependencies

* [CUDAv8](https://developer.nvidia.com/cuda-zone) and [cuDNNv6](https://developer.nvidia.com/cudnn) (or higher)
* [NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) (Compute Capability 6.1 or higher)

## Dev Dependencies

If you want to run the supervised or reinforcement learning programs to improve the quality of the weights or help development of the agent then you will need the following:

* [Python 3.6](https://www.python.org/) with [Tensorflow](https://tensorflow.org/)
* [Rust](https://www.rust-lang.org) (nightly)

## Training

To bootstrap the network from pre-generated data you will need an SGF file where each line contains a full game-tree, henceforth called *big SGF files*. If you do not have access to such a file you can use the `tools/sgf2big.py` tool to merge all SGF files contained within a directory to a single big SGF file. You may also want to do some data cleaning and balancing (to avoid bias in the value network) by removing duplicate games and ensuring we have the same amount of wins for both black and white.

```bash
./tools/sgf2big.py data/kgs/ > kgs_big.sgf
```

```bash
cat kgs_big.sgf | sort | uniq | shuf | ./tools/sgf2balance.py > kgs_bal.sgf
```

The moves contained within the file then needs to be pre-processed to a more appropriate format for training, this can be accomplished with the `--extract` command which takes the path to an SGF file and writes a binary representation of the features and the correct policy and winner for a random sub-set of the moves in the given big SGF file.

```bash
./dream_go --extract kgs_bal.sgf > kgs_big.bin
```

This binary file can then be feed into the bootstrap script which will tune the network weights to more accurately predict the moves played in the original SGF files. This script will run forever, so feel free to cancel it when you feel happy with the accuracy. You can monitor the accuracy (and a bunch of other stuff) using Tensorboard, whose logs are stored in the `logs/` directory. The final output will also be stored in the `models/` directory.

```bash
cd contrib/trainer
python -m dream_tf --start kgs_big.bin
```

```bash
tensorboard --logdir models/
```

When you are done training your network you need to transcode the weights from Tensorflow protobufs into a format that can be read by Dream Go, this can be accomplished using the `--dump` command of the bootstrap script:

```bash
python -m dream_tf --dump > dream-go.json
```

## Reinforcement Learning

Two reinforcement learning algorithms are supported by Dream Go. They differ only marginally in implementation but have vastly different hardware requirements. Which of the two algorithms is the best is currently unknown, but I would recommend _Expect Iteration_ because you most likely do not have the hardware requirements to run the _AlphaZero_ algorithm:

1. [AlphaZero](https://arxiv.org/abs/1712.01815)
1. [Expert Iteration](https://arxiv.org/abs/1705.08439)

### AlphaZero

If you want to use the [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm then you need to start by generating self-play games. The self-play games generated by _Dream Go_ are different from normal games played using the GTP interface in several ways, most notably they are more random (to encourage exploration, and avoid duplicate games), and a summary of the monte-carlo search tree is stored for each position. This monte-carlo summary is then used during training to expose a richer structure to the neural network.

This can be accomplished using the `--self-play` command-line option. I also recommend that you increase the `--num-threads` and `--batch-size` arguments for this since the defaults are tuned for the GTP interface which has different (real time) requirements. This program will generate 25,000 games (should take around 14 days on modern hardware):

```bash
./dream_go --num-threads 512 --batch-size 256 --self-play 25000 > self_play.sgf
```

The network should now be re-trained using this self-play, this is done in the same way as during the supervised training by first performing some basic data cleaning to avoid bias, converting the games to a binary representation and then training the network using tensorflow. You may wish to tune the `--num-samples` variable depending on how many self-play games you have generated as your goal should be to have around 2,000,000 examples for the neural network during training in total:

```bash
sort < self_play.sgf | uniq | shuf | ./tools/sgf2balance.py > self_play_bal.sgf
```
```bash
./dream_go --num-samples 80 --extract self_play_bal.sgf > self_play.bin
```
```bash
cd contrib/trainer/ && python3 -m dream_tf --start self_play.bin
```

One observation worth pointing out is that if we generate 25,000 games and each game contains on average 250 moves, we just generated 4,250,000 search _unnecessary_ trees. This is not quite true as they are still useful for the _value head_, but one could argue that we are **way** past the point of diminishing returns.

### Expert Iteration

The training procedure for [Expert Iteration](https://arxiv.org/abs/1705.08439) is almost the same as for _AlphaZero_ with two exceptions:

1. We generate `--policy-play` games instead of `--self-play` games. These are self-play games without any search, so they are about 800 to 1,600 times faster to generate, but of lower quality.
1. We generate the monte-carlo search tree during data extraction using the `--ex-it` switch only for examples that actually end-up as examples for the neural network.

```bash
./dream_go --num-games 256 --num-threads 512 --batch-size 256 --policy-play 1000000 > policy_play.sgf
```
```bash
sort < policy_play.sgf | uniq | shuf | ./tools/sgf2balance.py > policy_play_bal.sgf
```
```bash
./dream_go --num-games 256 --num-threads 512 --batch-size 256 --num-samples 2 --extract --ex-it policy_play_bal.sgf > policy_play.bin
```
```bash
cd contrib/trainer/ && python3 -m dream_tf --start policy_play.bin
```

For the values provided in this example, which generate 2,000,000 examples for the neural network it should take about 7 days to generate the required data (from 1,000,000 distinct games).

## Roadmap

* 1.0.0 - _Public Release_
* 0.7.0 - _Acceptance_
  * First version with a network trained from self-play games
* 0.6.0 - _Emergent_
  * Time and tournament commands for the GTP interface
  * Improved [neural network architecture](https://github.com/Chicoryn/dream-go/issues/25#issuecomment-377706857)
  * Improved performance with [DP4A](https://devblogs.nvidia.com/parallelforall/mixed-precision-programming-cuda-8/)
* 0.5.0 - _Assessment_
  * Optimize the monte carlo tree search parameters against other engines
  * Optimize neural network size for _best_ performance vs speed ratio
* 0.4.0 - _Awakening_
  * [GTP](http://www.lysator.liu.se/~gunnar/gtp/) interface
* 0.3.0 - _Slow-wave sleep_
  * Monte carlo tree search for self-play
* 0.2.0 - _Light Sleep_
  * Self-play agent without monte carlo tree search
  * Reinforcement learning using self-play games
* 0.1.0 - _Napping_
  * Supervised learning using a pre-existing dataset

## License

[Apache License 2.0](LICENSE)
