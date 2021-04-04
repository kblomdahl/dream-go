competition_type = 'playoff'
record_games = True
stderr_to_log = True

from glob import glob
from os.path import basename, isdir
from os import getcwd
import re
import subprocess

""" The directory that contains all engines to match against each other """
DIST = './engines/'

""" The number of rollouts to instruct each engine to use """
ROLLOUTS = 3200

""" The maximum number of matchups between each engine """
NUM_GAMES = 50

""" The number of games to aim for playing in total """
TOTAL_NUM_GAMES = 200


def leela(num_rollout):
    """ Returns an player that use of Leela 0.11.0 """
    return Player(
        'leela_gtp_opencl --noponder -g -p ' + str(num_rollout),
        is_reliable_scorer=True,
        startup_gtp_commands=[
            'time_settings 300 1 1'
        ]
    )

def dream_go_v050(path, num_rollout):
    environ = {'BATCH_SIZE': '16', 'NUM_ITER': str(num_rollout)}
    command = [path + '/dream_go']

    return Player(
        command,
        cwd=DIST,
        environ=environ,
        is_reliable_scorer=False,
        startup_gtp_commands=[]
    )

def dream_go(path, num_rollout):
    """ Returns an player that use Dream Go with the given binary """

    # make it output the _correct_ name from the GTP `name` command
    environ = {'DG_NAME': basename(path)}

    # build the docker container, and run the command from within the
    # container. This allows us to have **completely** different environments
    # for each engine.
    command = [
        'docker',
        'run',
        '--rm',  # do not keep container around
        '-i',  # interactive terminal
        '--sig-proxy=true',  # proxy all received signals
        '-v',  # mount 'engines' directory
        getcwd() + '/' + path + ':/app/engine'
    ]

    for key, value in environ.items():
        command += ['-e', key + '=' + str(value)]

    command += [
        subprocess.check_output('docker build -q ' + path, shell=True).strip(),
        '/app/engine/dream_go',
        '--no-ponder',
        '--num-rollout',
        str(num_rollout)
    ]

    return Player(
        command,
        cwd=None,
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

for path in glob(DIST + '/*'):
    name = basename(path)

    if isdir(path):
        if re.match(r'^dg-v050', name):
            players[name] = dream_go_v050(bin, ROLLOUTS)
        else:
            players[name] = dream_go(path, ROLLOUTS)

#
# Setup the gomill configuration
#
board_size = 19
komi = 7.5

names = list([name for name in players.keys()])
number_of_matchups = (len(names) * (len(names) - 1)) // 2
number_of_games = max(5, min(NUM_GAMES, TOTAL_NUM_GAMES // number_of_matchups))
matchups = list([
    Matchup(name_1, name_2, alternating=True, number_of_games=number_of_games)
    for (i, name_1) in enumerate(names)
    for name_2 in names
        if name_2 not in names[:i+1]
])
