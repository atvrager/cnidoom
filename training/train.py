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


def train(
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,
    cfg_path: str | None = None,
    resume: str | None = None,
    model_version: str = "v2",
    obs_h: int | None = None,
    obs_w: int | None = None,
    output: str = "doom_agent_ppo",
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

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=env)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device="auto",
            tensorboard_log=str(TB_LOG_DIR),
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
    )


if __name__ == "__main__":
    main()
