# Dream Go - All day, every day
Dream Go is an independent implementation of the algorithms and concepts presented by DeepMind in their [Master the Game of Go without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) paper with a few modifications to (maybe) make it feasible to develop a strong player without access to a supercomputer on the scale of [Sunway TaihuLight](https://en.wikipedia.org/wiki/Sunway_TaihuLight).

* Human games are used to bootstrap the network weights.
* Additional (synthetic) features inspired by [AlphaGo](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) and [DeepForest](https://arxiv.org/pdf/1511.06410.pdf) are used during training and inference.
* [Rapid Action Value Estimation (RAVE)](http://www.machinelearning.org/proceedings/icml2007/papers/387.pdf) is used during tree search as suggested by [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/pdf/1705.08439.pdf).

## Dependencies
* CUDAv8 and cuDNN

## Dev Dependencies
If you want to run the supervised or reinforcement learning programs to improve the quality of the weights or help development of the agent then you will need the following:

* [Python 3.6](https://www.python.org/) with [Tensorflow](https://tensorflow.org/)
* [Rust](https://www.rust-lang.org) (nightly)

## Training
To bootstrap the network from pre-generated data you will need a "big" SGF file, where each line contains a full game-tree. These moves then needs to be pre-processed to a more appropriate format for training, this can be accomplished with the `--dataset` command which takes the path to an SGF file and writes a binary representation of the features and the correct policy and winner for a random sub-set of the moves in the given SGF file.

```
$ cargo run --release -- --dataset kgs_big.sgf > kgs_big.bin
```

This binary file can then be feed into the bootstrap script which will tune the network weights to more accurately predict the moves from the original SGF file. This script will run forever, so feel free to cancel it when you feel happy with the accuracy. You can monitor the accuracy (and a bunch of other stuff) using Tensorboard, whose logs are stored in the `logs/` directory.

```
$ python tools/bootstrap.py kgs_big.bin
$ tensorboard --logdir logs/
```

## Roadmap
* 1.0.0 - _Public Release_
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
