# Trainer NG

The next generation of training script has two purposes, (i) upgrade to TensorFlow 2.0, an (ii) integrate the engine
into the training script, allowing for reinforcement learning without an external loop.

## Sequence diagram

```
+--------------+           +---------------+           +------------+      +-----------+        +---------+
|  Tensorflow  |           |  Keras Model  |           |  Dream Go  |      |  dg_mcts  |        |  dg_nn  |
+-------+------+           +-------+-------+           +-----+------+      +-----+-----+        +----+----+
        |                          +                         |                   +                   |
      +-+-+                                                +-+-+                                     |
      |   |              synchronize parameters            |   |                                     |
      |   +------------------------------------------------>   |                                   +-+-+
      |   |                                                |   |       update CPU parameters       |   |
      |   |                                                |   +----------------------------------->   |
      |   |                                                |   |                                   |   |
      |   |                                                |   |                                   +-+-+
      |   |                                                |   |                                     |
      |   |                                                |   |                                   +-+-+
      |   |                                                |   |          purge workspaces         |   |
      |   <------------------------------------------------+   +----------------------------------->   |
      |   |                       ok                       |   |                                   |   |
      |   |                                                +-+-+                 +                 +-+-+
      |   |                        |                         |                   |                   |
      |   |                      +-+-+                       |                   |                   |
      |   |                      |   |                       |                   |                   |
      |   +---------------------->   |                     +-+-+                 |                   |
      |   |                      |   |  generate examples  |   |                 |                   |
      |   |                      |   +---------------------+   |               +-+-+                 |
      |   |                      |   |                     |   |  policy play  |   |                 |
      |   |                      |   |                     |   +--------------->   |               +-+-+
      |   |                      |   |                     |   |               |   |    predict    |   |
      |   |                      |   |                     |   |               |   +--------------->   |
      |   |                      |   |                     |   |               |   |               |   |
      |   |                      |   |                     |   |               |   <---------------+   |
      |   |                      |   |                     |   |               |   |  predictions  |   |
      |   |                      |   |                     |   <---------------+   |               +-+-+
      |   |                      |   |                     |   |    example    |   |                 |
      |   |                      |   +---------------------+   |               +-+-+                 |
      |   |                      |   |      examples       |   |                 |                   |
      |   |                      |   |                     +-+-+                 |                   |
      |   |                      |   +-----+                 |                   |                   |
      |   |                      |   |     | training step   |                   |                   |
      |   |                      |   <-----+                 |                   |                   |
      |   |                      |   |                       |                   |                   |
      |   <----------------------+   |                       |                   |                   |
      |   |  updated parameters  |   |                       |                   |                   |
      +-+-+                      +-+-+                       |                   |                   |
        |                          |                         |                   |                   |
        +                          +                         +                   +                   +
```

## Keras Model

Recently a few changes to the original AlphaZero neural network architecture has become become increasingly more
popular. So I suggest that we embed them into the Dream Go architecture too:

- Additional training targets
- Squeeze Excitation Layer
- Global Average Pooling
- ResNeXt (???)

## Running

You need to build the Cython modules which links to Dream Go before you can run the training script:

```
make && python -m train_tf --start ...
```
