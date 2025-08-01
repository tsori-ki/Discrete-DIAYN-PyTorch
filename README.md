# Discrete-DIAYN-PyTorch

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This is a PyTorch implementation of **Discrete-DIAYN** (Diversity is All You Need) — a discrete action-space variant of the original DIAYN algorithm. It adapts DIAYN to work with environments where actions are **discrete**, enabling skill discovery without any extrinsic reward signal.

This project is forked from [alirezakazemipour/DIAYN-PyTorch](https://github.com/alirezakazemipour/DIAYN-PyTorch), which was designed for **continuous control** tasks using Soft Actor-Critic (SAC).

---

## Motivation

While intelligent agents in nature learn diverse and useful skills **without external rewards**, many RL algorithms rely on explicitly defined reward functions.

DIAYN reformulates this by maximizing **diversity** through skill discovery: the agent receives intrinsic rewards based on how **distinguishable** each skill is from others. In this discrete version, we adapt the same philosophy to environments with **discrete action spaces**.

The DIAYN reward is given by:

<p align="center">
  <img src="Results/equation.png" height=40>
</p>

where $z$ is the latent skill. The intrinsic reward encourages states that help a discriminator identify which skill was used.

---

## Features

* Skill discovery in **discrete action** environments
* Based on PyTorch & OpenAI Gym
* Compatible with simple control environments (e.g., `Acrobot`, `CartPole`)
* Modular design (agent, logger, networks)
* Checkpointing + RNG state saving

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
python3 main.py --mem_size=100000 --env_name="Acrobot-v1" --interval=50 --do_train --n_skills=10
```

### Options

```bash
optional arguments:
  --env_name ENV_NAME         Name of the environment
  --interval INTERVAL         Save/log every n episodes
  --do_train                  Train the agent (vs. test mode)
  --train_from_scratch        Start training from scratch
  --mem_size MEM_SIZE         Replay buffer size
  --n_skills N_SKILLS         Number of skills to learn
  --reward_scale REWARD_SCALE Reward scaling factor
  --seed SEED                 Random seed for reproducibility
```

To resume from checkpoints:

```bash
python3 main.py --env_name="Acrobot-v1" --mem_size=100000 --do_train --train_from_scratch
```

---

## Results

> x-axis = episode number

### Acrobot (n\_skills = 10)

<p align="center">
  <img src="Results/Acrobot/running_logq.png">
</p>

Emergent behaviors for different skills:

| Skill 1                      | Skill 2                      | Skill 3                      |
| ---------------------------- | ---------------------------- | ---------------------------- |
| ![](Gifs/Acrobot/skill1.gif) | ![](Gifs/Acrobot/skill2.gif) | ![](Gifs/Acrobot/skill3.gif) |

Reward distributions:
\| ![](Results/Acrobot/skill1.png) | ![](Results/Acrobot/skill2.png) | ![](Results/Acrobot/skill3.png) |

---

## Project Structure

```bash
├── Brain
│   ├── agent.py               # Discrete SAC agent with skill learning
│   ├── model.py               # Neural network modules
│   └── replay_memory.py       # Replay buffer
├── Common
│   ├── config.py              # Argument parser and default params
│   ├── logger.py              # Checkpointing, tensorboard logging
│   └── play.py                # Evaluation script
├── main.py                    # Main script to train/test the agent
├── requirements.txt
├── Results                    # Plots & visualizations
└── Gifs                       # Behavior GIFs for learned skills
```

---

## Dependencies

* `gym`
* `numpy`
* `torch`
* `tqdm`
* `opencv-python`

(see `requirements.txt` for exact versions)

---

## Reference

* **Original paper:** [*Diversity is All You Need: Learning Skills without a Reward Function* (Eysenbach et al., 2018)](https://arxiv.org/abs/1802.06070)
* [Original DIAYN-PyTorch repo (continuous)](https://github.com/alirezakazemipour/DIAYN-PyTorch)

---

## Acknowledgment

Thanks to the original authors and contributors:

* [@ben-eysenbach](https://github.com/ben-eysenbach)
* [@p-christ](https://github.com/p-christ)
* [@Dolokhow](https://github.com/Dolokhow)
* [@alirezakazemipour](https://github.com/alirezakazemipour)
