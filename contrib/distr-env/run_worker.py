#!/usr/bin/python3

import subprocess

from dg_storage import *
from shutil import copyfile
import re
import tempfile
import multiprocessing

RE = re.compile(r'RE\[([^\]]+)\]')

def score_game(sgf):
    """
    Returns the winner of the game in the given SGF file as
    judged by `gnugo`.
    """

    with tempfile.NamedTemporaryFile() as sgf_file:
        sgf_file.write(sgf.encode())
        sgf_file.flush()

        # start-up our judge (gnugo)
        gnugo = subprocess.Popen(
            ['/usr/games/gnugo',
             '--score', 'aftermath',
             '--chinese-rules', '--positional-superko',
             '-l', sgf_file.name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        for line in gnugo.stdout:
            line = line.decode('utf-8').strip()

            if 'White wins by' in line:  # White wins by 8.5 points
                return 'W+' + line.split()[3]
            elif 'Black wins by' in line:  # Black wins by 32.5 points
                return 'B+' + line.split()[3]

def clean_game(sgf):
    """ Returns the given game after it has been _cleaned up_. """

    winner = RE.search(sgf)
    resign = winner and 'R' in winner.group(1).upper()

    if winner and not resign:
        winner = score_game(sgf)

        if winner:
            sgf = re.sub(RE, 'RE[' + winner + ']', sgf)

    return sgf

# (1) download recent network
# (2) generate 1,000 fresh game records
if __name__ == '__main__':
    best_network = copy_most_recent_network()

    if best_network:
        copyfile(best_network, '/app/dream_go.json')

        game_records = ''
        env = { 'POLICY_ROLLOUT': '40' }
        proc = subprocess.Popen([
            '/app/dream_go',
            '--policy-play', '1000',
            '--num-rollout', '3200',
            '--num-samples', '1',
            '--ex-it'
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)

        with multiprocessing.Pool() as pool:
            def add_game_record(x):
                global game_records

                game_records += x
                game_records += '\r\n'

            # score the game as they get finished by the engine
            for line in proc.stdout:
                line = line.decode('utf-8').strip()
                pool.apply_async(clean_game, [line], callback=add_game_record)

            # wait for everything to finish
            _stdout, _stderr = proc.communicate()

            if proc.returncode != 0:
                quit(proc.returncode)

            pool.close()
            pool.join()

        upload_game_records(game_records, from_network=best_network, env=env, args=proc.args)
