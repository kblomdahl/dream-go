# `dream_go/database:0.5.0`

This image is responsible for storing the results produced by the other images
in a persistent manner. To do this it provides a _RESTful_ interface over HTTP
that can be easily accessed using command-line utilities, such as `curl`.

## Running

```bash
make docker
```
