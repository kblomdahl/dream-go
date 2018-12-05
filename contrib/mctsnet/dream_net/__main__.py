from dream_net.rules.color import opposite
from dream_net.sgf import one

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import dream_net.mcts_net as mcts_net
import numpy as np
import sys
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

# -------- Constants --------

NUM_ROLLOUTS = 10  # the number of MCTS rollouts to perform
BATCH_SIZE = 8  # the number of gradients to accumulate before applying them

tf.enable_eager_execution(
    config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    ),
    #device_policy=tf.contrib.eager.DEVICE_PLACEMENT_WARN,
    execution_mode=tf.contrib.eager.ASYNC
)

# -------- class MCTSNode --------

class MCTSNode:
    def __init__(self, statistics):
        self.statistics = statistics
        self.children = [None]*362

def probe(root, board, color):
    node_trace = []
    gradients = {}

    while True:
        # sample the (valid) one-hot action from the simulation policy
        with tf.GradientTape() as tape:
            pi = mcts_net.policy(root.statistics)
            log_pi = tf.log(pi)

        pi_vars = tape.watched_variables()
        pi_grad = tape.gradient(log_pi, pi_vars)

        if len(gradients) > 0:
            for key, grad in zip(pi_vars, pi_grad):
                gradients[key] += grad
        else:
            gradients = {key: grad for (key, grad) in zip(pi_vars, pi_grad)}

        # remove any illegal moves from the policy
        assert pi.shape == (1, 362), 'unrecognized policy shape {}'.format(pi.shape)

        pi = pi[0].numpy()

        for i in range(361):
            if not board.is_valid_aux(color, i):
                pi[i] = 0.0

        # update the board state with this new action
        pi_index = np.argmax(pi)
        if pi[pi_index] < 1e-8:
            break  # no valid moves

        if pi_index != 361:  # not pass
            board.place_aux(color, pi_index)

        # probe further, or terminate if we just expanded a leaf node.
        node_trace += (root,)

        if root.children[pi_index]:
            root = root.children[pi_index]
            color = opposite(color)
        else:
            break  # the child will be set in search_state

    return node_trace, gradients, pi_index, color


# -------- class SearchIteration --------

class SearchIteration:
    def __init__(self, losses, gradients, additional):
        self.losses = losses
        self.gradients = gradients
        self.additional = additional


def search_state(original_board, color, next1_value, next1_policy):
    root = MCTSNode(mcts_net.embed(original_board, color))

    for _m in range(NUM_ROLLOUTS):
        board = original_board.copy()
        node_trace, policy_gradients, next_index, next_color = probe(root, board, color)

        if len(node_trace) == 0:
            break

        with tf.GradientTape() as tape:
            # calculate the statistics for the new child
            h_1 = mcts_net.embed(board, next_color)

            node_trace[-1].children[next_index] = MCTSNode(h_1)

            # update the embeddings in the search tree from the bottom-up using
            # _backup network_.
            for node in reversed(node_trace):
                node.statistics = mcts_net.backup(node.statistics, h_1)
                h_1 = node.statistics

            # 
            readout_logits, readout_value = mcts_net.readout(root.statistics)
            loss_value = tf.squared_difference(next1_value, readout_value)
            loss_policy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=next1_policy,
                logits=readout_logits
            )

            loss = loss_value + loss_policy

        loss_vars = tape.watched_variables()
        loss_grad = tape.gradient(loss, loss_vars)
        loss_gradients = {key: grad for key, grad in zip(loss_vars, loss_grad)}

        yield SearchIteration({
            'total': loss,
            'value': loss_value,
            'policy': loss_policy
        }, {
            'readout': loss_gradients,
            'policy': policy_gradients
        }, {
            'logits': readout_logits,
            'value': readout_value
        })


def step_acc(board, next1_color, next1_value, next1_policy):
    """ """

    # perform the MCTSnet tree search
    results = list(search_state(board, next1_color, next1_value, next1_policy))

    # log summaries to TensorBoard
    next1_index = np.argmax(next1_policy)

    with tf.contrib.summary.record_summaries_every_n_global_steps(100):
        for key, value in results[-1].losses.items():
            tf.contrib.summary.scalar('losses/' + key, value)
            #tf.contrib.summary.histogram('progress/losses/' + key, [s_it[key] for s_it in results])

        with tf.device('/cpu:0'):
            tf.contrib.summary.scalar('accuracy/policy_1', tf.cast(tf.nn.in_top_k(results[-1].additional['logits'], [next1_index], 1), tf.float32))
            tf.contrib.summary.scalar('accuracy/policy_3', tf.cast(tf.nn.in_top_k(results[-1].additional['logits'], [next1_index], 3), tf.float32))
            tf.contrib.summary.scalar('accuracy/policy_5', tf.cast(tf.nn.in_top_k(results[-1].additional['logits'], [next1_index], 5), tf.float32))
        tf.contrib.summary.scalar('accuracy/value', tf.cast(tf.equal(tf.sign(results[-1].additional['value']), tf.sign(tf.cast(next1_value, tf.float32))), tf.float32))

    # calculate the REINFORCE policy gradient
    gradients = {}

    for search_it in results:
        # apply the _policy_ gradients:
        #
        # ??? Should we take the absolute value of this? How do we handle a
        #     final loss that is larger than the initial loss?
        R = search_it.losses['total'] - results[-1].losses['total']

        for key, grad in search_it.gradients['policy'].items():
            if grad is not None:
                gradients[key] = (R * grad + gradients[key]) if key in gradients else R * grad

        # apply the _readout_, _embedding_, and _backup_ gradients:
        for key, grad in search_it.gradients['readout'].items():
            if grad is not None:
                gradients[key] = (grad + gradients[key]) if key in gradients else grad

    # pretty-print the current board state
    final_readout = tf.nn.softmax(results[-1].additional['logits']).numpy()

    for i in range(361):
        if not board.is_valid_aux(next1_color, i):
            final_readout[0, i] = 0.0

    readout_index = np.argmax(final_readout[0])

    if next1_index == readout_index:
        marks = {index: mark for index, mark in [(next1_index, 'x')]}
    else:
        marks = {index: mark for index, mark in [
            (next1_index, 'x'),
            (readout_index, 'r')
        ]}

    print(board.to_string(marks))

    return gradients

def most_recent_model():
    """ Returns the directory in `models/` that is the most recent """
    import os

    all_models = ['models/' + m for m in os.listdir('models/')]

    return max(
        [m for m in all_models if os.path.isdir(m)],
        key=os.path.getmtime
    )

# 
with tf.device('/cpu:0'):
    global_step = tf.train.get_or_create_global_step()
    all_lines = tf.data.TextLineDataset(sys.argv[1:]).repeat()

# restore the most recent checkpoint
with tf.device('/gpu:0'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    checkpoint_vars = mcts_net.all_variables()
    checkpoint_vars['global_step'] = global_step
    checkpoint_vars['adam_optimizer'] = optimizer
    checkpoint = tfe.Checkpoint(**checkpoint_vars)

    try:
        recent_model_dir = most_recent_model()
        recent_model = recent_model_dir + '/ckpt'
        checkpoint_file = tf.train.latest_checkpoint(recent_model_dir)

        checkpoint.restore(checkpoint_file)
    except ValueError:
        from datetime import datetime

        now = datetime.now().strftime('%Y%m%d.%H%M')
        recent_model_dir = 'models/{}'.format(now)
        recent_model = recent_model_dir + '/ckpt'

# write TensorBoard statistics to the model directory
writer = tf.contrib.summary.create_file_writer(recent_model_dir)
writer.set_as_default()

# loop over all SGF files for all of eternity
num_steps = 0
gradients = {}

for line in all_lines:
    line = line.numpy().strip()

    try:
        # extract a single example from this game
        board, next1_color, next1_value, next1_policy = one(line)

        # probe into this game using a background thread, and then wait for a
        # previous job to finish (using a deterministic wait algorithm).
        with tf.device('/gpu:0'):
            for key, grad in step_acc(board, next1_color, next1_value, next1_policy).items():
                gradients[key] = gradients[key] + grad if key in gradients else grad

        num_steps += 1

        # apply the gradients if we have collected a full batches worth of
        # gradients.
        if num_steps % BATCH_SIZE == 0:
            grads_and_vars = list([(grad, var) for var, grad in gradients.items()])

            with tf.device('/gpu:0'):
                optimizer.apply_gradients(
                    grads_and_vars=grads_and_vars,
                    global_step=global_step
                )

            gradients = {}

        if num_steps % 1000 == 0:
            checkpoint.save(recent_model)
    except ValueError:
        pass # skip invalid examples
