competition_type = 'playoff'
record_games = True
stderr_to_log = True

from glob import glob
from os.path import basename
from os import access, X_OK

""" The directory that contains all engines to match against each other """
DIST = '/app/dist/'

""" The number of rollouts to instruct each engine to use """
NUM_ROLLOUT = 800

""" The players to generate match-ups for, if `None` then matchups are generated
for all engines """
MAIN_PLAYERS = sorted([
    engine
    for engine in glob(DIST + '/*')
    if access(engine, X_OK)
])

def dream_go(name, num_rollout=None, is_reliable_scorer=True):
    if num_rollout is None:
        num_rollout = NUM_ROLLOUT

    # make it output the _correct_ name from the GTP `name` command
    environ = {'DG_NAME': basename(name), 'CUDA_VISIBLE_DEVICES': '0,1'}
    command = [name, '--num-rollout', str(num_rollout)]

    return Player(
        command,
        cwd=DIST,
        environ=environ,
        is_reliable_scorer=is_reliable_scorer,
        startup_gtp_commands=[
            #'kgs-time_settings byoyomi 0 1 1'
        ]
    )

""" All players that will participate in this playoff """
players = { basename(name): dream_go(name) for name in MAIN_PLAYERS }

#
# Setup the gomill configuration
#
board_size = 19
komi = 7.5

names = list(players.keys())
matchups = list([
    Matchup(name_1, name_2, alternating=True, number_of_games=200)
    for (i, name_1) in enumerate(names)
    for name_2 in names
        if name_2 not in names[:i+1]
])
