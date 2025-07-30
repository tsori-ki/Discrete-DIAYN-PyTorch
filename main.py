import gymnasium as gym
from Brain import SACAgent
from Common import Play, Logger
import numpy as np
from tqdm import tqdm
import torch


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n, dtype=np.float32)
    z_one_hot[z_] = 1.0
    return np.concatenate([s, z_one_hot])


#  1.  Parameters (edit here or make them arguments)
params = {
    "mem_size":        1_000_000,
    "env_name":        "Acrobot-v1",
    "interval":        100,
    "do_train":        False,          # ← set False to play
    "train_from_scratch": True,
    "reward_scale":    1.0,
    "seed":            123,
    "lr":              3e-4,
    "batch_size":      256,
    "max_n_episodes":  2_000,
    "max_episode_len": 1_000,
    "gamma":           0.99,
    "alpha":           0.1,
    "tau":             0.005,
    "n_hiddens":       300,
    "n_skills":        20,
}

#  2.  Infer state / action dimensions from env
probe_env = gym.make(params["env_name"])
params["n_states"] = probe_env.observation_space.shape[0]
params["n_actions"] = probe_env.action_space.n

probe_env.close()

#  3.  Create env, agent, logger, etc.
env = gym.make(params["env_name"], render_mode="rgb_array")
p_z = np.full(params["n_skills"], 1.0 / params["n_skills"], dtype=np.float32)

agent  = SACAgent(p_z=p_z, **params)
logger = Logger(agent, **params)

#  4.  TRAIN MODE
if params["do_train"]:

    if not params["train_from_scratch"]:
        start_ep, last_logq_zs, torch_rng, random_rng = logger.load_weights()
        agent.hard_update_target_network()
        agent.set_rng_states(torch_rng, random_rng)
    else:
        start_ep, last_logq_zs = 0, 0.0
        np.random.seed(params["seed"])

    logger.on()

    for episode in tqdm(range(start_ep + 1, params["max_n_episodes"] + 1)):
        z = np.random.choice(params["n_skills"], p=p_z)
        state, _ = env.reset(seed=params["seed"] + episode)
        state = concat_state_latent(state, z, params["n_skills"])

        episode_reward, logq_vals = 0.0, []
        max_steps = min(params["max_episode_len"], env.spec.max_episode_steps)

        for step in range(1, max_steps + 1):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = concat_state_latent(next_state, z, params["n_skills"])
            agent.store(state, z, done, action, next_state)

            logq = agent.train()
            logq_vals.append(last_logq_zs if logq is None else logq)

            episode_reward += reward
            state = next_state
            if done:
                break

        logger.log(episode,
                   episode_reward,
                   z,
                   float(np.mean(logq_vals)),
                   step,
                   *agent.get_rng_states())

    env.close()

#  5.  PLAY MODE
else:
    logger.load_weights()
    player = Play(env, agent, n_skills=params["n_skills"])
    player.evaluate()