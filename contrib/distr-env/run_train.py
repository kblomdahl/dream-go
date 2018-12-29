#!/usr/bin/python

import petname
import subprocess

from dg_storage import *

# (1) download recent models and game records.
# (2) train a new model
best_model = copy_most_recent_model()
next_model = petname.Generate(3, '_')

if best_model:
    args = [
        '/usr/bin/python',
        '-m', 'dream_tf',
        '--start', '--warm-start', best_model,
        '--steps', '40960000',
        '--name', next_model
    ] + copy_most_recent_games()
else:
    args = [
        '/usr/bin/python',
        '-m', 'dream_tf',
        '--start',
        '--name', next_model
    ] + copy_most_recent_games()

proc = subprocess.run(args, stderr=subprocess.DEVNULL)
if proc.returncode != 0:
    quit(proc.returncode)

upload_next_model(next_model)

# dump the network weights from the trained model
proc = subprocess.run([
    '/usr/bin/python',
    '-m', 'dream_tf',
    '--dump'
], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

if proc.returncode != 0:
    quit(proc.returncode)

upload_next_network(next_model, proc.stdout, args=args)

# wait until all of the networks has been evaluated and
# assigned an ELO score
# wait_until_all_models_rated()
