# Gomill

An docker environment that can be used to run tournaments between different
versions of Dream Go.

## Dependencies

- Docker

## Running

Copy the engines, weights, and dockerfile that you want to match between to the
`engines/` folder, together with their weights, and then run:

```bash
pip install gomill
ringmaster scripts/tournament.ctl.py run
```
