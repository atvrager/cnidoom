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
from training.model import DoomFeatureExtractor

CHECKPOINT_DIR = Path("checkpoints")
TB_LOG_DIR = Path("tb_doom")


def make_env(cfg_path: str | None = None, frame_skip: int = 4):
    def _init():
        return DoomHybridEnv(cfg_path=cfg_path, frame_skip=frame_skip)

    return _init


def train(
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,
    cfg_path: str | None = None,
    resume: str | None = None,
) -> PPO:
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=DoomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[64], vf=[64]),
    )

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(
            resume,
            env=make_vec_env(
                make_env(cfg_path),
                n_envs=n_envs,
                vec_env_cls=SubprocVecEnv,
            ),
        )
    else:
        env = make_vec_env(
            make_env(cfg_path),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )
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
    eval_env = make_vec_env(make_env(cfg_path), n_envs=1)

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
    model.save("doom_agent_ppo")
    print("Training complete. Model saved to doom_agent_ppo.zip")

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
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        cfg_path=args.cfg,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
