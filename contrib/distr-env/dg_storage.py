#!/usr/bin/python

from base64 import standard_b64encode
from crc32c import crc32
from datetime import datetime
from google.cloud import storage
from os.path import basename, dirname, isfile
from os import makedirs, getenv
from glob import glob
from time import sleep
import json
import petname
import subprocess
import struct
import sys

storage_client = storage.Client()
bucket_name = 'dream-go'
bucket = storage_client.lookup_bucket(bucket_name)
if not bucket:
    bucket = storage_client.create_bucket(bucket_name)

def get_most_recent_model():
    try:
        most_recent_file = max(
            [blob for blob in bucket.list_blobs(prefix='models/') if blob.size > 0],
            key=lambda blob: ((blob.metadata or {}).get('elo', 0.0), blob.time_created)
        )

        if most_recent_file:
            return dirname(most_recent_file.name)
    except ValueError:
        return None

def get_most_recent_network():
    try:
        return max(
            [blob for blob in bucket.list_blobs(prefix='networks/') if blob.size > 0],
            key=lambda blob: ((blob.metadata or {}).get('elo', 0.0), blob.time_created)
        )
    except ValueError:
        return None

def blob_already_exists(blob, dest_file):
    if isfile(dest_file):
        with open(dest_file, 'rb') as file:
            raw_crc = crc32(file.read())

        encoded = standard_b64encode(struct.pack('>I', raw_crc)).decode('ascii')

        return encoded == blob.crc32c
    return False

def copy_most_recent_model():
    """ Copy the most recent model to the 'models/' directory """

    best_model = get_most_recent_model()

    if best_model:
        print('Warm-starting from {}'.format(best_model), end='', flush=True)

        for blob in bucket.list_blobs(prefix=best_model):
            dest_file = 'models/{}/{}'.format(
                basename(best_model),
                basename(blob.name)
            )

            if not blob_already_exists(blob, dest_file):
                makedirs(dirname(dest_file), exist_ok=True)
                with open(dest_file, 'wb') as file:
                    print('.', end='', flush=True)

                    blob.download_to_file(file)

        print()

    return best_model

def copy_most_recent_network():
    best_network = get_most_recent_network()

    if best_network:
        dest_file = 'networks/{}'.format(
            basename(best_network.name)
        )

        if not blob_already_exists(best_network, dest_file):
            with open(dest_file, 'wb') as file:
                best_network.download_to_file(file)

        return dest_file
    else:
        return None

def wait_until_all_models_rated():
    """ Wait until all models has been assigned an ELO score. """

    while True:
        models = {}

        for blob in bucket.list_blobs(prefix='models/'):
            if blob.size > 0:
                models[dirname(blob.name)] = True

                if blob.metadata and 'elo' in blob.metadata:
                    return True

        if len(models) <= 1:
            return True

        sleep(600)  # 10 minutes

def copy_most_recent_games():
    """ Download the 100,000 most recent games, each file should
    contain 1,000 game records. So we need to download the 100
    most recent files. """

    files = []
    blobs = sorted(
        [blob for blob in bucket.list_blobs(prefix='games/') if blob.size > 0],
        key=lambda blob: blob.time_created
    )

    print('Loading training data...', end='', flush=True)

    for blob in blobs[-200:]:
        dest_file = 'data/{}'.format(basename(blob.name))
        files += (dest_file,)

        if not blob_already_exists(blob, dest_file):
            with open(dest_file, 'wb') as file:
                print('.', end='', flush=True)

                blob.download_to_file(file)

    print('', flush=True)

    return files

def upload_next_model(next_model):
    """ Upload the specified model to google storage. """

    for src_file in glob('models/*{}/*'.format(next_model)):
        print('Uploading', src_file)

        blob = bucket.blob(src_file)
        blob.upload_from_filename(filename=src_file)

def upload_next_network(next_model, data, args=None):
    """ Upload the specified network to google storage. """

    blob = bucket.blob('networks/{}.json'.format(next_model))
    blob.metadata = {
        'args': json.dumps(args, sort_keys=True),
        'rev': getenv('GIT_REV')
    }

    blob.upload_from_string(data, 'application/json')

def upload_game_records(data, from_network=None, env=None, args=None):
    """ Upload the specified game records to google storage. """

    dest_file = 'games/{}.sgf'.format(
        datetime.now().strftime('%Y%m%d.%H%M')
    )
    print('Uploading', dest_file)

    blob = bucket.blob(dest_file);
    blob.metadata = {
        'args': json.dumps(args, sort_keys=True),
        'env': json.dumps(env, sort_keys=True),
        'network': from_network,
        'rev': getenv('GIT_REV')
    }

    blob.upload_from_string(data, 'application/x-go-sgf')
