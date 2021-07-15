import queue

from botbowlcurriculum import make_academy
import numpy as np
from common.env_wrappers import BatchedEnvironment
from r2d2_main import make_ffai_env, create_agent
import gym

import atari_py
from seed_rl.common import utils

import tensorflow as tf

batch_size = 4


def get_envs_epsilon(env_ids, num_training_envs, num_eval_envs, eval_epsilon):
  """Per-environment epsilon as in Apex and R2D2.

  Args:
    env_ids: <int32>[inference_batch_size], the environment task IDs (in range
      [0, num_training_envs+num_eval_envs)).
    num_training_envs: Number of training environments. Training environments
      should have IDs in [0, num_training_envs).
    num_eval_envs: Number of evaluation environments. Eval environments should
      have IDs in [num_training_envs, num_training_envs + num_eval_envs).
    eval_epsilon: Epsilon used for eval environments.

  Returns:
    A 1D float32 tensor with one epsilon for each input environment ID.
  """
  # <float32>[num_training_envs + num_eval_envs]
  epsilons = tf.concat(
      [tf.math.pow(0.4, tf.linspace(1., 8., num=num_training_envs)),
       tf.constant([eval_epsilon] * num_eval_envs)],
      axis=0)
  return tf.gather(epsilons, env_ids)


def apply_epsilon_greedy(actions, env_ids, num_training_envs,
                         num_eval_envs, eval_epsilon, action_mask):
  """Epsilon-greedy: randomly replace actions with given probability.

  Args:
    actions: <int32>[batch_size] tensor with one action per environment.
    env_ids: <int32>[inference_batch_size], the environment task IDs (in range
      [0, num_envs)).
    num_training_envs: Number of training environments.
    num_eval_envs: Number of eval environments.
    eval_epsilon: Epsilon used for eval environments.
    action_mask: <bool>[batch_size, num_actions]: action mask for each environment

  Returns:
    A new <int32>[batch_size] tensor with one action per environment. With
    probability epsilon, the new action is random, and with probability (1 -
    epsilon), the action is unchanged, where epsilon is chosen for each
    environment.
  """
  batch_size = tf.shape(actions)[0]
  epsilons = get_envs_epsilon(env_ids, num_training_envs, num_eval_envs,
                              eval_epsilon)

  def call_on_me(mask):
      allowed = tf.cast(tf.squeeze(tf.where(mask), axis=1), dtype=tf.int32)
      rand_int = tf.random.uniform((1,), 0, len(allowed), dtype=tf.int32)[0]
      return allowed[rand_int]

  random_actions = tf.map_fn(call_on_me, action_mask, fn_output_signature=tf.int32)

  probs = tf.random.uniform(shape=[batch_size])
  return tf.where(tf.math.less(probs, epsilons), random_actions, actions)

env = make_ffai_env(None, None)
spat_obs, non_spat_obs, action_mask = env.reset()

env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec(spat_obs.shape, spat_obs.dtype, 'spat_obs'),
      tf.TensorSpec(non_spat_obs.shape, non_spat_obs.dtype, 'nonspat_obs'),
      tf.TensorSpec(action_mask.shape, action_mask.dtype, 'action_mask'),
      tf.TensorSpec([], tf.bool, 'abandoned'),
      tf.TensorSpec([], tf.int32, 'episode_step'),
  )
agent = create_agent(env_output_specs, env.action_space.n)
del env

# Create environment
envs = BatchedEnvironment(make_ffai_env, batch_size, 0, None)

# Initialize academy
academy = make_academy()
lecture_probs_and_levels = academy.get_probs_and_levels()
lecture_outcome_queue = queue.Queue(5000)

# Setup first EnvOutput
spat_obs, non_spat_obs, action_mask = envs.reset(lecture_probs_and_levels)
reward = np.zeros(batch_size, np.float32)
raw_reward = np.zeros(batch_size, np.float32)
done = np.zeros(batch_size, np.bool)
abandoned = np.zeros(batch_size, np.bool)
episode_step = np.zeros(batch_size, np.int32)

# Setup input to agent
agent_state = agent.initial_state(batch_size)
prev_actions = tf.zeros((batch_size,), dtype=tf.uint8)


unroll_size = 250
spat_obs_unroll = np.zeros([unroll_size] + list(spat_obs.shape))
non_spat_obs_unroll = np.zeros([unroll_size] + list(non_spat_obs.shape))
action_mask_unroll = np.zeros([unroll_size] + list(action_mask.shape))
reward_unroll = np.zeros([unroll_size] + list(reward.shape))
done_unroll = np.zeros([unroll_size] + list(done.shape))
abandoned_unroll = np.zeros([unroll_size] + list(abandoned.shape))
episode_step_unroll = np.zeros([unroll_size] + list(episode_step.shape))
prev_actions_unroll = np.zeros([unroll_size] + list(prev_actions.shape))

for i in range(unroll_size):
    env_outputs = utils.EnvOutput(reward, done, spat_obs, non_spat_obs, action_mask, abandoned, episode_step)

    spat_obs_unroll[i] = spat_obs
    non_spat_obs_unroll[i] = non_spat_obs
    action_mask_unroll[i] = action_mask
    reward_unroll[i] = reward
    done_unroll[i] = done
    abandoned_unroll[i] = abandoned
    episode_step_unroll[i] = episode_step

    input_ = (prev_actions, env_outputs)

    # __call__()
    actions, agent_state = agent(input_, agent_state)


    #print(actions.action)
    random_actions = apply_epsilon_greedy(actions.action, [0] * batch_size, batch_size, 0, 0.1, action_mask)
    actions = actions._replace(action=random_actions)
    #print(actions.action)

    # step
    rewards, done, _, _, _ = envs.step(actions.action.numpy())

    spat_obs, non_spat_obs, action_mask = envs.reset_if_done(done, lecture_probs_and_levels)


    assert all([env.lecture is not None for env in envs.envs])
    if done.any():
        print("done and reset!")
    outcomes = np.stack([env.get_lecture_outcome() for env in envs.envs])

    academy.log_training(outcomes)


env_outputs = utils.EnvOutput(reward_unroll, done_unroll, spat_obs_unroll, non_spat_obs_unroll, action_mask_unroll, abandoned_unroll, episode_step_unroll)
input_ = (prev_actions_unroll, env_outputs)

# __call__()
actions, agent_state = agent(input_, agent_state, unroll=True)

print("Passed the test!")
