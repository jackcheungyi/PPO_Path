from PIL import Image
import numpy as np 
import cv2 
import random 
from Env_wrapper_mlp import PlanEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
import time 

MODEL_PATH=r"trained_models_mlp_model_20240527110409/ppo_plan_1375000_steps"

im = Image.open("HKSB_6F.pgm")
im_array = np.asarray(im)
im_down = cv2.pyrDown(im_array)
im_down = cv2.pyrDown(im_down)

seed = random.randint(0, 1e9)

env = env = PlanEnv(seed=seed,map=im_down,silent_mode=False)

model = MaskablePPO.load(MODEL_PATH)

# obs,_ = env.reset()
episode = 0
done = False 
while 1 :
    seed = random.randint(0, 1e9)
    env = env = PlanEnv(seed=seed,map=im_down,silent_mode=False)
    obs,_ = env.reset(seed=seed)
    done = False
    info = None
    print(f"=================== Episode {episode + 1} ==================")
    while not done :
        action_masks= get_action_masks(env)
        action, _ = model.predict(obs,action_masks=action_masks)

        # prev_mask = env.get_action_mask()

        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)

    env.close()
    episode = episode +1 
    time.sleep(5)