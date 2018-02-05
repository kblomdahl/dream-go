# Docker images

This directory contains some docker images that can be used to more easily
deploy _Dream Go_ (and keep stuff organized). All three images are necessary to
create the closed loop, but some steps are more compute intense than others, so
I would recommend you try the following deployments:

- 1x Database (my trivial database implementation does not scale horizontally)
- 10x GPU Worker (as many as you can)
- 1x Tensorflow trainer

## `dream_go/db:0.5.0` - Database image

This image is responsible for storing all of self-play games, as well as any
extracted features from said self-play games and the final weights themselves.
This image is mandatory for any of the other images to work as they use it to
retrieve their initial state, and dump their final result into it.

To run this image you need to setup the following:

- Mount some persistent storage to `/mnt/dream_db/`.

## `dream_go/worker:0.5.0` - GPU Worker

This image is responsible for generating the self-play games from the latest
weights stored in the database image. It will run an infinite loop where each
iteration generate a small batch of self-play games using the latest weights and
then upload the games and their features to the database.

To run this image you need to setup the following:

- Set the `DB` environment variable to point towards the location of the
  database. For example `upload.dg.io:8080`.

## `dream_go/trainer:0.5.0` - Tensorflow trainer

This image is responsible for retrieving features from the database and
generating new weights from them. It will retrieve the 500,000 most recent
features from the database and train a neural network for 51500 steps before
uploading the new weights to the database.

You can monitor the training process by connecting to port 6006 of this image,
which expose a Tensorboard instance with some interesting metrics.

To run this image you need to setup the following:

- Set the `DB` environment variable to point towards the location of the
  database. For example `upload.dg.io:8080`.

Optionally:

- Mount some persistent storage to `/app/logs` if you want to save any generated
  logs from the training.
