competition_type = 'playoff'
record_games = True
stderr_to_log = True

from glob import glob
from os.path import basename
import re

""" The directory that contains all engines to match against each other """
DIST = '/app/dist/'

def dream_go(name='dg-v061', num_rollout=None, is_reliable_scorer=True):
    # make it output the _correct_ name from the GTP `name` command
    environ = {'DG_NAME': '%s-%04d' % (name, num_rollout), 'CUDA_VISIBLE_DEVICES': '1'}
    command = [DIST + name, '--no-ponder', '--num-rollout', str(num_rollout)]

    return Player(
        command,
        cwd=DIST,
        environ=environ,
        is_reliable_scorer=is_reliable_scorer,
        startup_gtp_commands=[]
    )

""" All players that will participate in this playoff """
players = {
    '1':    dream_go(num_rollout=   1),
    '10':   dream_go(num_rollout=  10),
    '40':   dream_go(num_rollout=  40),
    '100':  dream_go(num_rollout= 100),
    '200':  dream_go(num_rollout= 200),
    '400':  dream_go(num_rollout= 400),
    '800':  dream_go(num_rollout= 800),
    '1600': dream_go(num_rollout=1600),
    '3200': dream_go(num_rollout=3200),
}

#
# Setup the gomill configuration
#
board_size = 19
komi = 7.5

matchups = list([
    Matchup(   '1',   '10', alternating=True, number_of_games=50),
    Matchup(  '10',   '40', alternating=True, number_of_games=50),
    Matchup(  '40',  '100', alternating=True, number_of_games=50),
    Matchup( '100',  '200', alternating=True, number_of_games=50),
    Matchup( '200',  '400', alternating=True, number_of_games=50),
    Matchup( '400',  '800', alternating=True, number_of_games=50),
    Matchup( '800', '1600', alternating=True, number_of_games=50),
    Matchup('1600', '3200', alternating=True, number_of_games=50),
])
