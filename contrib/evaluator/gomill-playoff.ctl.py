#
# Setup the gomill configuration
#
competition_type = 'playoff'
record_games = True
stderr_to_log = True
board_size = 19
komi = 7.5

#
# Setup the `matchups`
#
from glob import glob
from os.path import basename
import re

""" All players that will participate in this playoff """
players = {}

for bin in glob('./dist/*'):
    name = basename(bin)

    if re.match(r'[a-z_]+$', name):
        players[name] = Player(
            [bin, '--num-rollout', '1'],
            startup_gtp_commands=[
                "kgs-time_settings byoyomi 0 5 1"  # 5 seconds byo-yomi
            ],
            environ={'DG_NAME': name}
        )

names = sorted([name for name in players.keys()])
matchups = list([
    Matchup(name_1, name_2, alternating=True, number_of_games=20)
    for (i, name_1) in enumerate(names)
    for name_2 in names[i+1:]
])
