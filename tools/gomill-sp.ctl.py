competition_type = 'playoff'
record_games = True
stderr_to_log = True

from glob import glob
from os.path import basename
import re

""" The directory that contains all engines to match against each other """
DIST = '/home/kalle/Documents/Code/dream-go/dist/'

""" The number of rollouts to instruct each engine to use """
ROLLOUTS = 1

""" The players to generate match-ups for, if `None` then matchups are generated
for all engines """
MAIN_PLAYERS = ['dg-19x256', 'dg-v050']


def leela(num_rollout):
    """ Returns an player that use of Leela 0.11.0 """
    return Player(
        'leela_gtp_opencl --noponder -g -p ' + str(num_rollout),
        is_reliable_scorer=True
    )

def dream_go(bin, num_rollout, is_reliable_scorer):
    """ Returns an player that use Dream Go with the given binary """

    name = basename(bin)

    if name == 'dg-v050':
        environ = {'BATCH_SIZE': '16', 'NUM_ITER': str(num_rollout)}
        command = [bin]
    else:
        # make it output the _correct_ name from the GTP `name` command
        environ = {'DG_NAME': name}
        command = [bin, '--batch-size', '16', '--num-rollout', str(num_rollout)]

    return Player(
        command,
        cwd=DIST,
        environ=environ,
        is_reliable_scorer=True
    )

def by_version(name):
    """ Returns the given name with any version number zero padded to a
    _safe_ length. """
    if name.startswith('dg-'):
        version = name[3:]

        try:
            return 'dg-%08d' % (int(version))
        except:
            return name
    else:
        return name

""" All players that will participate in this playoff """
players = {
    'leela': leela(ROLLOUTS),
}

for bin in glob(DIST + 'dg-*'):
    name = basename(bin)

    if re.match(r'dg-v[0-9]+$', name):
        # released version, always include in the playoff
        players[name] = dream_go(bin, ROLLOUTS, name != 'dg-v050')
    elif re.match(r'dg-[0-9]+$', name):
        # reinforcement learning revision
        players[name] = dream_go(bin, ROLLOUTS, True)
    elif re.match(r'dg-[^\.]+$', name):
        # other?
        players[name] = dream_go(bin, ROLLOUTS, True)

#
# Setup the gomill configuration
#
board_size = 19
komi = 7.5

names = sorted([name for name in players.keys()], key=by_version)
if not MAIN_PLAYERS:
    MAIN_PLAYERS = names

matchups = list([
    Matchup(name_1, name_2, alternating=True, number_of_games=50)
    for (i, name_1) in enumerate(MAIN_PLAYERS)
    for name_2 in names
        if name_2 not in MAIN_PLAYERS[:i+1]
])
