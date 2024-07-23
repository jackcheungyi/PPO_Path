import os 
import sys 
import random 
from PIL import Image
import numpy as np 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from Env_warpper_mlp import PlanEnv
import cv2 
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)
from stable_baselines3.common.callbacks import BaseCallback
import pygame

class RenderCallback(BaseCallback):
    def __init__(self, env):
        super(RenderCallback, self).__init__()
        self.env = env

    def _on_step(self) -> bool:
        self.env.render()
        pygame.time.wait(50)
        return True




NUM_ENV = 11
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
# Set the save directory




# log_file_path = os.path.join(save_dir, "training_log.txt")
# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler



def make_env(seed=0):
    def _init():
        im = Image.open("HKSB_6F.pgm")
        im_array = np.asarray(im)
        im_down = cv2.pyrDown(im_array)
        im_down = cv2.pyrDown(im_down)
        env = PlanEnv(seed=seed,map=im_down,silent_mode=True)
        # env = ActionMasker(env, PlanEnv.get_action_mask)
        # env = ActionMasker(env,env.get_wrapper_attr('get_action_mask'))
        env = Monitor(env)
        # env.reset(seed)
        return env
    return _init


def main():
    print("hello")
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])
    # im = Image.open("HKSB_6F.pgm")
    # im_array = np.asarray(im)
    # im_down = cv2.pyrDown(im_array)
    # im_down = cv2.pyrDown(im_down)
    # seed = random.randint(0, 1e9)
    # env = PlanEnv(seed=seed,map=im_down,silent_mode=False)

    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

     # # Instantiate a PPO agent
    model = MaskablePPO(
        # MaskableActorCriticPolicy,
        # 'CnnPolicy',
        'MlpPolicy',
        env,
        device="cuda",
        verbose=1,
        n_steps=1024,
        # n_steps = 10240,
        batch_size=512,
        n_epochs=10,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR
       
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = f"trained_models_mlp_model_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625*4 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_plan")
    render_callback = RenderCallback(env)
    # with open(log_file_path, 'w') as log_file:
    #     sys.stdout = log_file

    #     model.learn(
    #         total_timesteps=int(100000000),
    #         callback=[checkpoint_callback],
    #         progress_bar=True
    #     )
    #     env.close()

    
     

    model.learn(
        total_timesteps=int(100000000),
        # callback=[checkpoint_callback,render_callback],
        callback=[checkpoint_callback],
        progress_bar=True
    )
    env.close()

    # Restore stdout
    # sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_plan_final.zip"))

if __name__ == "__main__":
    main()
