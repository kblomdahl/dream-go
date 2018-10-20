# Gomill

An docker environment that can be used to run tournaments between different versions of Dream Go.

## Running

Copy the engines that you want to match between to the `engines/` folder, together with their weights, and then run:

```bash
./start_dev_container.sh
ringmaster scripts/tournament.ctl.py run
```
