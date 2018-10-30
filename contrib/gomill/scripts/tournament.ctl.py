competition_type = 'playoff'
record_games = True
stderr_to_log = True

from glob import glob
from os.path import basename
import re

""" The directory that contains all engines to match against each other """
DIST = '/app/engines/'

""" The number of rollouts to instruct each engine to use """
ROLLOUTS = 3200

""" The number of matchups between each engine """
NUM_GAMES = 50


def leela(num_rollout):
    """ Returns an player that use of Leela 0.11.0 """
    return Player(
        'leela_gtp_opencl --noponder -g -p ' + str(num_rollout),
        is_reliable_scorer=True,
        startup_gtp_commands=[
            'time_settings 300 1 1'
        ]
    )

def dream_go_v050(bin, num_rollout):
    environ = {'BATCH_SIZE': '16', 'NUM_ITER': str(num_rollout)}
    command = [bin]

    return Player(
        command,
        cwd=DIST,
        environ=environ,
        is_reliable_scorer=False,
        startup_gtp_commands=[]
    )

def dream_go(bin, num_rollout):
    """ Returns an player that use Dream Go with the given binary """

    # make it output the _correct_ name from the GTP `name` command
    environ = {'DG_NAME': basename(bin)}
    command = [bin, '--no-ponder', '--num-rollout', str(num_rollout)]

    return Player(
        command,
        cwd=DIST,
        environ=environ,
        is_reliable_scorer=True,
        startup_gtp_commands=[
            #'time_settings 300 1 1'
        ]
    )

""" All players that will participate in this playoff """
players = {
    #'leela': leela(ROLLOUTS),
}

for bin in glob(DIST + '/*'):
    name = basename(bin)

    if re.match(r'.*\.json', name):
        pass
    elif re.match(r'^dg-v050', name):
        players[name] = dream_go_v050(bin, ROLLOUTS)
    else:
        players[name] = dream_go(bin, ROLLOUTS)

#
# Setup the gomill configuration
#
board_size = 19
komi = 7.5

names = list([name for name in players.keys()])
matchups = list([
    Matchup(name_1, name_2, alternating=True, number_of_games=NUM_GAMES)
    for (i, name_1) in enumerate(names)
    for name_2 in names
        if name_2 not in names[:i+1]
])
