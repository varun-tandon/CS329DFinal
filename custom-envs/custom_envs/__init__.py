from gymnasium.envs.registration import register

register(
    id="custom_envs/CartPole-v1",
    entry_point="custom_envs.envs:CartPoleEnv",
    max_episode_steps=500,
)
