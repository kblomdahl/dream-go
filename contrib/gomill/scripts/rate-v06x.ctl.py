competition_type = 'playoff'
record_games = True
stderr_to_log = True

from glob import glob
from os.path import basename
import re

""" The directory that contains all engines to match against each other """
DIST = '/app/dist/'

""" The number of rollouts to instruct each engine to use """
NUM_ROLLOUT = 3200

""" The players to generate match-ups for, if `None` then matchups are generated
for all engines """
MAIN_PLAYERS = [
    'dg-v060',
    'dg-v061-newly_wanted_walrus',
    'dg-v061-really_heroic_kite',
    'dg-v061-truly_happy_akita',
    'dg-v061-wholly_deep_guinea',
    'dg-v061-wholly_witty_tarpon',
]

def dream_go(name, num_rollout=None, is_reliable_scorer=True):
    if num_rollout is None:
        num_rollout = NUM_ROLLOUT

    # make it output the _correct_ name from the GTP `name` command
    environ = {'DG_NAME': name, 'CUDA_VISIBLE_DEVICES': '0,1'}
    command = [DIST + name, '--no-ponder', '--num-rollout', str(num_rollout)]

    return Player(
        command,
        cwd=DIST,
        environ=environ,
        is_reliable_scorer=is_reliable_scorer,
        startup_gtp_commands=[]
    )

""" All players that will participate in this playoff """
players = { name: dream_go(name) for name in MAIN_PLAYERS }

#
# Setup the gomill configuration
#
board_size = 19
komi = 7.5

matchups = list([
    Matchup(name_1, name_2, alternating=True, number_of_games=50)
    for (i, name_1) in enumerate(MAIN_PLAYERS)
    for name_2 in MAIN_PLAYERS
        if name_2 not in MAIN_PLAYERS[:i+1]
])
