# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""SEED agent using Keras."""

import collections

import numpy as np
from seed_rl.common import utils
import tensorflow as tf

AgentOutput = collections.namedtuple('AgentOutput', 'action q_values')
AgentState = collections.namedtuple(
    # core_state: the opaque state of the recurrent core of the agent.
    # frame_stacking_state: a list of the last (stack_size - 1) observations
    #   with shapes typically [batch_size, height, width, 1].
    'AgentState', 'core_state frame_stacking_state')

STACKING_STATE_DTYPE = tf.int32


def initial_frame_stacking_state(stack_size, batch_size, observation_shape):
  """Returns the initial frame stacking state.

  It should match what stack_frames accepts and produces.

  Args:
    stack_size: int, the number of frames that should be stacked to form the
      observation provided to the neural network. stack_size=1 corresponds to no
      stacking.
    batch_size: int tensor.
    observation_shape: list, shape of a single observation, e.g.
      [height, width, 1].

  Returns:
    <STACKING_STATE_DTYPE>[batch_size, prod(observation_shape)] or an empty
    tuple if stack_size=1.
  """
  if stack_size == 1:
    return ()
  return tf.zeros(
      tf.concat([[batch_size], [tf.math.reduce_prod(observation_shape)]],
                axis=0),
      dtype=STACKING_STATE_DTYPE)


def stack_frames(frames, frame_stacking_state, done, stack_size):
  """Stacks frames.

  The [height, width] center dimensions of the tensors below are typical, but
  could be anything else consistent between tensors.

  Args:
    frames: <float32>[time, batch_size, height, width, channels]. These should
      be un-normalized frames in range [0, 255]. channels must be equal to 1
      when we actually stack frames (stack_size > 1).
    frame_stacking_state: If stack_size > 1, <int32>[batch_size, height*width].
      () if stack_size=1.
      Frame are bit-packed. The LSBs correspond to the oldest frames, MSBs to
      newest. Frame stacking state contains un-normalized frames in range
      [0, 256). We use [height*width] for the observation shape instead of
      [height, width] because it speeds up transfers to/from TPU by a factor ~2.
    done: <bool>[time, batch_size]
    stack_size: int, the number of frames to stack.
  Returns:
    A pair:
      - stacked frames, <float32>[time, batch_size, height, width, stack_size]
        tensor (range [0, 255]). Along the stack dimensions, frames go from
        newest to oldest.
      - New stacking state with the last (stack_size-1) frames.
  """
  if frames.shape[0:2] != done.shape[0:2]:
    raise ValueError(
        'Expected same first 2 dims for frames and dones. Got {} vs {}.'.format(
            frames.shape[0:2], done.shape[0:2]))
  batch_size = frames.shape[1]
  obs_shape = frames.shape[2:-1]
  if stack_size > 4:
    raise ValueError('Only up to stack size 4 is supported due to bit-packing.')
  if stack_size > 1 and frames.shape[-1] != 1:
    raise ValueError('Due to frame stacking, we require last observation '
                     'dimension to be 1. Got {}'.format(frames.shape[-1]))
  if stack_size == 1:
    return frames, ()
  if frame_stacking_state[0].dtype != STACKING_STATE_DTYPE:
    raise ValueError('Expected dtype {} got {}'.format(
        STACKING_STATE_DTYPE, frame_stacking_state[0].dtype))

  frame_stacking_state = tf.reshape(
      frame_stacking_state, [batch_size] + obs_shape)

  # Unpacked 'frame_stacking_state'. Ordered from oldest to most recent.
  unstacked_state = []
  for i in range(stack_size - 1):
    # [batch_size, height, width]
    unstacked_state.append(tf.cast(tf.bitwise.bitwise_and(
        tf.bitwise.right_shift(frame_stacking_state, i * 8), 0xFF),
                                   tf.float32))

  # Same as 'frames', but with the previous (stack_size - 1) frames from
  # frame_stacking_state prepended.
  # [time+stack_size-1, batch_size, height, width, 1]
  extended_frames = tf.concat(
      [tf.reshape(frame, [1] + frame.shape + [1])
       for frame in unstacked_state] +
      [frames],
      axis=0)

  # [time, batch_size, height, width, stack_size].
  # Stacked frames, but does not take 'done' into account. We need to zero-out
  # the frames that cross episode boundaries.
  # Along the stack dimensions, frames go from newest to oldest.
  stacked_frames = tf.concat(
      [extended_frames[stack_size - 1 - i:extended_frames.shape[0] - i]
       for i in range(stack_size)],
      axis=-1)

  # We create a mask that contains true when the frame should be zeroed out.
  # Setting the final shape of the mask early actually makes computing
  # stacked_done_masks a few times faster.
  done_mask_row_shape = done.shape[0:2] + [1] * (frames.shape.rank - 2)
  done_masks = [
      tf.zeros(done_mask_row_shape, dtype=tf.bool),
      tf.reshape(done, done_mask_row_shape)
  ]
  while len(done_masks) < stack_size:
    previous_row = done_masks[-1]
    # Add 1 zero in front (time dimension).
    done_masks.append(
        tf.math.logical_or(
            previous_row,
            tf.pad(previous_row[:-1],
                   [[1, 0]] + [[0, 0]] * (previous_row.shape.rank - 1))))

  # This contains true when the frame crosses an episode boundary and should
  # therefore be zeroed out.
  # Example: ignoring batch_size, if done is [0, 1, 0, 0, 1, 0], stack_size=4,
  # this will be:
  # [[0 0, 0, 0, 0, 0],
  #  [0 1, 0, 0, 1, 0],
  #  [0 1, 1, 0, 1, 1],
  #  [0 1, 1, 1, 1, 1]].T
  # <bool>[time, batch_size, 1, 1, stack_size].
  stacked_done_masks = tf.concat(done_masks, axis=-1)
  stacked_frames = tf.where(
      stacked_done_masks,
      tf.zeros_like(stacked_frames), stacked_frames)

  # Build the new bit-packed state.
  # We construct the new state from 'stacked_frames', to make sure frames
  # before done is true are zeroed out.
  # This shifts the stack_size-1 items of the last dimension of
  # 'stacked_frames[-1, ..., :-1]'.
  shifted = tf.bitwise.left_shift(
      tf.cast(stacked_frames[-1, ..., :-1], tf.int32),
      # We want to shift so that MSBs are newest frames.
      [8 * i for i in range(stack_size - 2, -1, -1)])
  # This is really a reduce_or, because bits don't overlap.
  new_state = tf.reduce_sum(shifted, axis=-1)

  new_state = tf.reshape(new_state, [batch_size, obs_shape.num_elements()])

  return stacked_frames, new_state


def _unroll_cell(inputs, done, start_state, zero_state, recurrent_cell):
  """Applies a recurrent cell on inputs, taking care of managing state.

  Args:
    inputs: A tensor of shape [time, batch_size, <remaining dims>]. These are
      the inputs passed to the recurrent cell.
    done: <bool>[time, batch_size].
    start_state: Recurrent cell state at the beginning of the input sequence.
      Opaque tf.nest structure of tensors with batch front dimension.
    zero_state: Blank recurrent cell state. The current recurrent state will be
      replaced by this blank state whenever 'done' is true. Same shape as
      'start_state'.
    recurrent_cell: Function that will be applied at each time-step. Takes
      (input_t: [batch_size, <remaining dims>], current_state) as input, and
      returns (output_t: [<cell output dims>], new_state).

  Returns:
    A pair:
      - The time-stacked outputs of the recurrent cell. Shape [time,
        <cell output dims>].
      - The last state output by the recurrent cell.
  """
  stacked_outputs = []
  state = start_state
  inputs_list = tf.unstack(inputs)
  done_list = tf.unstack(done)
  assert len(inputs_list) == len(done_list), (
      "Inputs and done tensors don't have same time dim {} vs {}".format(
          len(inputs_list), len(done_list)))
  # Loop over time dimension.
  # input_t: [batch_size, batch_size, <remaining dims>].
  # done_t: [batch_size].
  for input_t, done_t in zip(inputs_list, done_list):
    # If the episode ended, the frame state should be reset before the next.
    state = tf.nest.map_structure(
        lambda x, y, done_t=done_t: tf.where(
            tf.reshape(done_t, [done_t.shape[0]] + [1] *
                       (x.shape.rank - 1)), x, y),
        zero_state,
        state)
    output_t, state = recurrent_cell(input_t, state)
    stacked_outputs.append(output_t)
  return tf.stack(stacked_outputs), state


class DuelingLSTMDQNNet_ffai(tf.Module):
  """The recurrent network used to compute the agent's Q values.

  This is the dueling LSTM net similar to the one described in
  https://openreview.net/pdf?id=rkHVZWZAZ (only the Q(s, a) part), with the
  layer sizes mentioned in the R2D2 paper
  (https://openreview.net/pdf?id=r1lyTjAqYX), section Hyper parameters.
  """

  def __init__(self, num_actions, observation_shape, stack_size=1):
    super(DuelingLSTMDQNNet_ffai, self).__init__(name='dueling_lstm_dqn_net_ffai')
    self._num_actions = num_actions
    self._body_spat = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, [3, 3], 1,
                               padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], 1,
                               padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, [3, 3], 1,
                               padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, [3, 3], 1,
                               padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
    ])
    self._body_nonspat = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu')
    ])

    self._value = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', name='hidden_value'),
        tf.keras.layers.Dense(1, name='value_head'),
    ])
    self._advantage = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', name='hidden_advantage'),
        tf.keras.layers.Dense(self._num_actions, use_bias=False,
                              name='advantage_head'),
    ])
    self._core = tf.keras.layers.LSTMCell(1)
    self._observation_shape = observation_shape
    self._stack_size = stack_size

  def initial_state(self, batch_size):
    return AgentState(
        core_state=self._core.get_initial_state(
            batch_size=batch_size, dtype=tf.float32),
        frame_stacking_state=initial_frame_stacking_state(
            self._stack_size, batch_size, self._observation_shape))

  def _torso(self, env_output):
    # [batch_size, output_units]
    conv_out = self._body_spat(env_output[2]) #spatial obs
    nonspat_out = self._body_nonspat(env_output[3]) #"non spatial obs"

    # [batch_size, num_actions]
    # one_hot_prev_action = tf.one_hot(prev_action, self._num_actions)
    # [batch_size, torso_output_size]
    #return tf.concat(
    #    [conv_out, nonspat_out, tf.expand_dims(env_output.reward, -1), one_hot_prev_action],
    #    axis=1)

    return tf.concat(
        [conv_out, nonspat_out],
        axis=1)

  def _head(self, core_output, env_output):
    # [batch_size, 1]
    value = self._value(core_output)

    # [batch_size, num_actions]
    advantage = self._advantage(core_output)
    advantage -= tf.reduce_mean(advantage, axis=-1, keepdims=True)

    # [batch_size, num_actions]
    q_values = value + advantage

    q_values = tf.where(env_output[4], q_values, -9e9) #-np.Inf)

    action = tf.cast(tf.argmax(q_values, axis=1), tf.int32)
    return AgentOutput(action, q_values)

  def __call__(self, input_, agent_state, unroll=False):
    """Applies a network mapping observations to actions.

    Args:
      input_: A pair of:
        - previous actions, <int32>[batch_size] tensor if unroll is False,
          otherwise <int32>[time, batch_size].
        - EnvOutput, where each field is a tensor with added front
          dimensions [batch_size] if unroll is False and [time, batch_size]
          otherwise.
      agent_state: AgentState with batched tensors, corresponding to the
        beginning of each unroll.
      unroll: should unrolling be aplied.

    Returns:
      A pair of:
        - outputs: AgentOutput, where action is a tensor <int32>[time,
            batch_size], q_values is a tensor <float32>[time, batch_size,
            num_actions]. The time dimension is not present if unroll=False.
        - agent_state: Output AgentState with batched tensors.
    """
    if not unroll:
      # Add time dimension.
      input_ = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                     input_)
    prev_actions, env_outputs = input_
    outputs, agent_state = self._unroll(prev_actions, env_outputs, agent_state)
    if not unroll:
      # Remove time dimension.
      outputs = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

    return outputs, agent_state

  def _unroll(self, prev_actions, env_outputs, agent_state):
    # [time, batch_size, <field shape>]
    rewards, dones, spat_obs, non_spat_obs, action_mask, _, _ = env_outputs
    spat_obs = tf.cast(spat_obs, tf.float32)
    non_spat_obs = tf.cast(non_spat_obs, tf.float32)
    action_mask = tf.cast(action_mask, tf.bool)
    dones = tf.cast(dones, tf.bool)
    rewards = tf.cast(rewards, tf.bool)


    initial_agent_state = self.initial_state(batch_size=tf.shape(dones)[1])

    #stacked_frames, frame_state = stack_frames(
    #    observation, agent_state.frame_stacking_state, done, self._stack_size)

    env_outputs = rewards, dones, spat_obs, non_spat_obs, action_mask


    # [time, batch_size, torso_output_size]
    torso_outputs = utils.batch_apply(self._torso, (env_outputs, ))

    #core_outputs, core_state = _unroll_cell(
    #    torso_outputs, dones, agent_state.core_state,
    #    initial_agent_state.core_state,
    #    self._core)

    agent_output = utils.batch_apply(self._head, (torso_outputs, env_outputs))
    #agent_output = utils.batch_apply(self._head, (core_outputs, env_outputs))
    # return agent_output, AgentState(core_state, frame_state) # Original

    return agent_output, initial_agent_state #AgentState(core_state, ())
