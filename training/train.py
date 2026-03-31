"""PPO training script for the Doom agent."""

import os

# Prevent protobuf MessageFactory errors from tensorflow/tensorboard.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from training.env import DoomHybridEnv
from training.model import DoomFeatureExtractor, DoomFeatureExtractorV2

CHECKPOINT_DIR = Path("checkpoints")
TB_LOG_DIR = Path("tb_doom")


MODEL_VERSIONS = {
    "baseline": {
        "extractor": DoomFeatureExtractor,
        "features_dim": 256,
        "net_arch": dict(pi=[64], vf=[64]),
        "obs_h": 45,
        "obs_w": 60,
    },
    "v2": {
        "extractor": DoomFeatureExtractorV2,
        "features_dim": 128,
        "net_arch": dict(pi=[64], vf=[64]),
        "obs_h": 60,
        "obs_w": 80,
    },
}


def make_env(
    cfg_path: str | None = None,
    frame_skip: int = 4,
    obs_h: int = 45,
    obs_w: int = 60,
):
    def _init():
        return DoomHybridEnv(
            cfg_path=cfg_path, frame_skip=frame_skip, obs_h=obs_h, obs_w=obs_w
        )

    return _init


def scaled_ppo_kwargs(n_envs: int) -> dict:
    """Return PPO hyperparameters scaled for the number of environments.

    At 8 envs (baseline), we use the original values.  As envs increase,
    batch_size scales up so the number of gradient steps per rollout
    stays roughly constant, and n_epochs decreases to avoid over-fitting
    on each rollout.

    Rollout buffer = n_envs × n_steps samples.
    Gradient steps per update = (buffer / batch_size) × n_epochs.

    Target: ~2500 gradient steps per update (same as 8 envs baseline).
    """
    n_steps = 2048
    buffer = n_envs * n_steps

    if n_envs <= 16:
        batch_size = 64
        n_epochs = 10
    elif n_envs <= 64:
        batch_size = 512
        n_epochs = 6
    else:
        # 128+ envs: large rollouts, scale batch, fewer epochs.
        batch_size = 2048
        n_epochs = 4

    grad_steps = (buffer // batch_size) * n_epochs
    print(
        f"PPO scaling: {n_envs} envs × {n_steps} steps = {buffer:,} buffer, "
        f"batch={batch_size}, epochs={n_epochs}, "
        f"~{grad_steps:,} grad steps/update"
    )

    return dict(
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )


def train(
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,
    cfg_path: str | None = None,
    resume: str | None = None,
    model_version: str = "v2",
    obs_h: int | None = None,
    obs_w: int | None = None,
    output: str = "doom_agent_ppo",
    batch_size: int | None = None,
    n_epochs: int | None = None,
) -> PPO:
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    ver = MODEL_VERSIONS[model_version]
    obs_h = obs_h or ver["obs_h"]
    obs_w = obs_w or ver["obs_w"]

    policy_kwargs = dict(
        features_extractor_class=ver["extractor"],
        features_extractor_kwargs=dict(features_dim=ver["features_dim"]),
        net_arch=ver["net_arch"],
    )

    env = make_vec_env(
        make_env(cfg_path, obs_h=obs_h, obs_w=obs_w),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Scale PPO hyperparameters for the env count.
    ppo_kw = scaled_ppo_kwargs(n_envs)
    if batch_size is not None:
        ppo_kw["batch_size"] = batch_size
    if n_epochs is not None:
        ppo_kw["n_epochs"] = n_epochs

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=env)
        # Apply scaled hyperparameters to the resumed model.
        model.batch_size = ppo_kw["batch_size"]
        model.n_epochs = ppo_kw["n_epochs"]
        model.n_steps = ppo_kw["n_steps"]
        # Trigger internal buffer reallocation for new n_steps/n_envs.
        model._setup_model()
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device="auto",
            tensorboard_log=str(TB_LOG_DIR),
            **ppo_kw,
        )

    # Eval env (single, for speed).
    eval_env = make_vec_env(make_env(cfg_path, obs_h=obs_h, obs_w=obs_w), n_envs=1)

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=str(CHECKPOINT_DIR),
            name_prefix="doom_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(CHECKPOINT_DIR / "best"),
            eval_freq=max(25_000 // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(output)
    print(f"Training complete. Model saved to {output}.zip")

    # Close environments so VizDoom subprocesses don't linger.
    env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Doom PPO agent")
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--envs", type=int, default=8, help="Number of parallel environments"
    )
    parser.add_argument("--cfg", type=str, default=None, help="VizDoom config file")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v2",
        choices=list(MODEL_VERSIONS.keys()),
        help="Model architecture version (default: v2)",
    )
    parser.add_argument(
        "--obs-res",
        type=str,
        default=None,
        help="Observation resolution as HxW, e.g. '60x80' (overrides model default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="doom_agent_ppo",
        help="Output checkpoint path (without .zip extension)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override PPO batch size (default: auto-scaled from --envs)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Override PPO epochs per update (default: auto-scaled from --envs)",
    )
    args = parser.parse_args()

    obs_h = None
    obs_w = None
    if args.obs_res:
        obs_h, obs_w = (int(x) for x in args.obs_res.split("x"))

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        cfg_path=args.cfg,
        resume=args.resume,
        model_version=args.model_version,
        obs_h=obs_h,
        obs_w=obs_w,
        output=args.output,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )


if __name__ == "__main__":
    main()
