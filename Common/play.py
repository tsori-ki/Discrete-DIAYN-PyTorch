import mujoco
import mujoco.viewer
import cv2
import numpy as np
import os
import gymnasium as gym


class Play:
    def __init__(self, env, agent, n_skills):
        self.env = env
        self.agent = agent
        self.n_skills = n_skills
        self.agent.set_policy_net_to_cpu_mode()
        self.agent.set_policy_net_to_eval_mode()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Vid/"):
            os.mkdir("Vid/")

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate(self):
        for z in range(self.n_skills):
            video_writer = cv2.VideoWriter(f"Vid/skill{z}" + ".avi", self.fourcc, 50.0, (250, 250))
            s = self.env.reset()
            # Handle both old and new gym API
            if isinstance(s, tuple):
                s = s[0]  # New gym API returns (obs, info)
            
            s = self.concat_state_latent(s, z, self.n_skills)
            episode_reward = 0
            
            # Handle different ways to get max episode steps
            max_steps = getattr(self.env.spec, 'max_episode_steps', 1000)
            if max_steps is None:
                max_steps = 1000
                
            for _ in range(max_steps):
                action = self.agent.choose_action(s)
                step_result = self.env.step(action)
                
                # Handle both old and new gym API
                if len(step_result) == 4:
                    s_, r, done, _ = step_result
                else:  # New gym API returns 5 values
                    s_, r, terminated, truncated, _ = step_result
                    done = terminated or truncated
                    
                s_ = self.concat_state_latent(s_, z, self.n_skills)
                episode_reward += r
                
                if done:
                    break
                    
                s = s_
                
                # Render the environment
                try:
                    # Try the standard render method first
                    I = self.env.render()
                    if I is None:
                        # If render returns None, try with mode parameter
                        I = self.env.render(mode='rgb_array')
                except:
                    # Fallback for environments that still use the old API
                    I = self.env.render(mode='rgb_array')
                
                if I is not None:
                    # Convert from RGB to BGR for OpenCV
                    if len(I.shape) == 3 and I.shape[2] == 3:
                        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                    I = cv2.resize(I, (250, 250))
                    cv2.putText(I, f"Skill: {z}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    video_writer.write(I)
                    
            print(f"skill: {z}, episode reward:{episode_reward:.1f}")
            video_writer.release()
            
        self.env.close()
        cv2.destroyAllWindows()