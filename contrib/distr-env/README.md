# Distributed Training Environment

This is the training environment for the reinforcement learning process that is used to improve the network weights. It is based on:

- Google Storage
- Tensorflow

## Running

Copy the `google-storage-auth.json` file to the current directory. You will need this to login and upload files to google storage:

```bash
make && ./run.sh
```
