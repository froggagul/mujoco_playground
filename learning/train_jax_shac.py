# Copyright 2025 DeepMind Technologies Limited
#
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
# ==============================================================================
"""Train a SHAC agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
# from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.agents.ppo import networks_vision as ppo_networks_vision
# from brax.training.agents.ppo import train as ppo

from shac import networks as shac_networks
from shac import train as shac

from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import tensorboardX
import wandb
from flax import linen


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.INFO)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_ACTOR_LEARNING_RATE = flags.DEFINE_float("actor_learning_rate", 5e-4, "Actor learning rate")
_CRITIC_LEARNING_RATE = flags.DEFINE_float("critic_learning_rate", 5e-4, "Critic learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)

# SHAC specific parameters
_TAU = flags.DEFINE_float(
    "tau", 0.005, "Target network update rate (1-alpha from the original paper)"
)
_LAMBDA = flags.DEFINE_float(
    "lambda", 0.95, "GAE lambda parameter for advantage estimation"
)
_TD_LAMBDA = flags.DEFINE_boolean(
    "td_lambda", True, "Use TD(lambda) for critic updates"
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  from ml_collections import config_dict
  from mujoco_playground._src import locomotion, dm_control_suite

  if env_name == "G1JoystickFlatTerrain":
    env_config = locomotion.get_default_config(env_name)
    rl_config = config_dict.create(
        num_timesteps=100_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        # SHAC specific parameters
        tau=0.005,  # 1-alpha from the original paper
        lambda_=0.85,  # GAE lambda parameter
        td_lambda=True,  # Use TD(lambda) for critic updates
        num_resets_per_eval=1,  # Number of resets per evaluation
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(128, 128, 128, 128),
            value_hidden_layer_sizes=(256, 256, 256, 256, 256),
            policy_obs_key="state",
            value_obs_key="state",
        ),
    )
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 20
    rl_config.entropy_cost = 0.005
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  elif env_name == "CartpoleBalance":
    env_config = dm_control_suite.get_default_config(env_name)
    rl_config = config_dict.create(
        num_timesteps=60_000_000, # 5000 episodes
        num_evals=1000, # evaluation number (epoch)
        num_eval_envs=64,
        num_resets_per_eval=0,  # Number of resets per evaluation

        reward_scaling=10.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,

        # training configuration
        num_envs=64,
        unroll_length=32, # rollout length per policy, short horizon length
        batch_size=256, # batch_size

        num_critic_update=16, # update number of critic
        num_critic_minibatch_per_update=4, # number of minibatches when per critic update

        discounting=0.99,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        entropy_cost=0.0,

        alpha=0.2,
        lambda_=0.95,  # TD lambda parameter
        td_lambda=True,  # Use TD(lambda) for critic updates
    )
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(64, 64),
        value_hidden_layer_sizes=(64, 64),
        # activation=linen.elu,
    )
  else:
    raise ValueError(f"not implemented {env_name}")

  return rl_config


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)

  shac_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    shac_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    shac_params.num_timesteps = 0
  if _NUM_EVALS.present:
    shac_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    shac_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    shac_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    shac_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    shac_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    shac_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    shac_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    shac_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _ACTOR_LEARNING_RATE.present:
    shac_params.actor_learning_rate = _ACTOR_LEARNING_RATE.value
  if _CRITIC_LEARNING_RATE.present:
    shac_params.critic_learning_rate = _CRITIC_LEARNING_RATE.value
  if _ENTROPY_COST.present:
    shac_params.entropy_cost = _ENTROPY_COST.value
  if _DISCOUNTING.present:
    shac_params.discounting = _DISCOUNTING.value
  if _NUM_ENVS.present:
    shac_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    shac_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    shac_params.batch_size = _BATCH_SIZE.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    shac_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    shac_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    shac_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    shac_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  env = registry.load(_ENV_NAME.value, config=env_cfg)
  if _RUN_EVALS.present:
    shac_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    shac_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    shac_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  # SHAC specific parameters
  if _TAU.present:
    shac_params.tau = _TAU.value
  if _LAMBDA.present:
    shac_params.lambda_ = _LAMBDA.value
  if _TD_LAMBDA.present:
    shac_params.td_lambda = _TD_LAMBDA.value

  print(f"Environment Config:\n{env_cfg}")
  print(f"SHAC Training Parameters:\n{shac_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs/shac").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.init(project="mjxrl", entity="froggagul", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(shac_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = shac_networks.make_shac_networks
  if hasattr(shac_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **shac_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = shac_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      shac.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  times = [time.monotonic()]

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  # Load evaluation environment.
  config_overrides = {"impl": _IMPL.value}
  eval_env = None
  if not _VISION.value:
    eval_env = registry.load(
        _ENV_NAME.value, config=env_cfg, config_overrides=config_overrides
    )
  num_envs = 1
  if _VISION.value:
    num_envs = env_cfg.vision_config.render_batch_size

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(
          _ENV_NAME.value, config=env_cfg, config_overrides=config_overrides
      )
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=shac_params.episode_length,
          action_repeat=shac_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        shac_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Run evaluation rollouts.
  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = eval_env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      if _VISION.value:
        traj_data = jax.tree_util.tree_map(lambda x: x[0], traj_data)
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)

  # def get_grad_of_reward_wrt_action(rng, state):
  #   def reward_fn(action):
  #     next_state = eval_env.step(state, action)
  #     return next_state.reward

  #   action = jit_inference_fn(state.obs, rng)[0]
  #   grad_fn = jax.grad(reward_fn)
  #   action_grad = grad_fn(action)
  #   return action, action_grad

  # action, action_grad = jax.jit(jax.vmap(get_grad_of_reward_wrt_action))(
  #   rng, reset_states
  # )
  # print(action_grad)
  # breakpoint()

  # Render and save the rollout.
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, scene_option=scene_option, camera="lookatcart",
    )
    media.write_video(f"shac_rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as 'shac_rollout{i}.mp4'.")


if __name__ == "__main__":
  app.run(main)
