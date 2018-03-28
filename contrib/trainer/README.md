# `dream_go/trainer:0.5.0`

This image is responsible for optimizing the model weights to better predict the
moves suggested by `dream_go/worker`. The image will do this by reading examples
from the `features` collection, and then writing the final weights into the
`weights` collection.

You can monitor the optimization process by running:

```bash
tensorboard --logdir models/
```

## Running

```bash
make docker DEVICE=0 DB=x.y.z.w:8080
```
