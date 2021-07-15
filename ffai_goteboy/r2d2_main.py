"""R2D2 binary for FFAI by Mattias Bermell.

Actor and learner are in the same binary so that all flags are shared.
"""

from absl import app
from absl import flags

import tensorflow as tf
from botbowlcurriculum import all_lectures
from ffai import GotebotWrapper

try:
    from seed_rl.agents.r2d2 import learner  # imports grpc which doesn't work outside of docker
    from seed_rl.common import actor  # imports grpc which doesn't work outside of docker
except tf.errors.NotFoundError:
    print("Failed to import learner and actor! Continuing anyway")

from seed_rl.ffai_goteboy import networks as ffai_networks

from seed_rl.common import common_flags

import ffai
import gym

FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')

flags.DEFINE_integer('stack_size', 1, 'Number of frames to stack.')


def create_agent(env_output_specs, num_actions):
    return ffai_networks.DuelingLSTMDQNNet_ffai(num_actions, env_output_specs.spat_obs.shape)


# Use the same
def create_optimizer(unused_final_iteration):
    learning_rate_fn = lambda iteration: FLAGS.learning_rate
    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
                                         epsilon=FLAGS.adam_epsilon)
    return optimizer, learning_rate_fn


def make_ffai_env(task, config):
    #env =  gym.make("FFAI-wrapped-v3")

    env = GotebotWrapper(gym.make("FFAI-v3"), all_lectures)

    print(" ---- MADE AN ENVIRONEMENT! ---- ")

    env.observation_space.dtype = tf.float32
    #env.config.pathfinding_enabled = True
    return env


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if FLAGS.run_mode == 'actor':
        actor.actor_loop(make_ffai_env)
    elif FLAGS.run_mode == 'learner':
        learner.learner_loop(make_ffai_env,
                             create_agent,  # TODO - correct agent
                             create_optimizer)
    else:
        raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
    app.run(main)
