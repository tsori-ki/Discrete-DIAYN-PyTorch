import gymnasium as gym
import numpy as np
from tqdm import tqdm

from Brain.agent import SAC  # your Discrete SAC now
from Common import Play, Logger


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = {
        "mem_size": 1000000,
        "env_name": "Acrobot-v1",
        "interval": 100,
        "do_train": True,
        "n_skills": 10,
        "train_from_scratch": True,
        "reward_scale": 1,
        "seed": 123,
        "lr": 0.0003,
        "batch_size": 256,
        "max_n_episodes": 1000,
        "max_episode_len": 500,
        "gamma": 0.99,
        "alpha": 0.1,
        "tau": 0.005,
        "n_hiddens": 128,
        "fixed_network_update_freq": 1000,

    }

    # Initialize env to get state/action spaces
    test_env = gym.make(params["env_name"])
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.n  # ✅ DISCRETE
    test_env.close()

    params.update({
        "n_states": n_states,
        "n_actions": n_actions,
        "state_shape": n_states + params["n_skills"],
    })
    print("params:", params)

    # Use render_mode only if evaluating
    env = gym.make(params["env_name"])

    # Skill prior
    p_z = np.full(params["n_skills"], 1 / params["n_skills"])

    # Initialize agent + logger
    agent = SAC(p_z=p_z, **params)
    print("device = ", agent.device)
    logger = Logger(agent, **params)

    if params["do_train"]:
        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            agent.set_rng_states(torch_rng_state, random_rng_state)
            print("Continue training.")
        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state, _ = env.reset(seed=params["seed"] + episode)
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            for step in range(1, 1 + params["max_episode_len"]):
                action = agent.choose_action(state)  # returns discrete int
                next_state, reward, done, _, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, action, reward, next_state, done)

                logq_zs = agent.train()
                logq_zses.append(last_logq_zs if logq_zs is None else logq_zs)

                episode_reward += reward
                state = next_state

                if done:
                    break

            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       *agent.get_rng_states(),
                       )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
