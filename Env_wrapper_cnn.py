from plan_game import PlanGame
# import gym 
import gymnasium as gym
import numpy as np 
import cv2
from typing import TYPE_CHECKING, Optional
from typing import List
from stable_baselines3.common.env_checker import check_env
class PlanEnv(gym.Env):
    def __init__(self, seed = 0, map = [],silent_mode = True, limit_step =True):
        super().__init__()
        self.game = PlanGame(seed=seed, map=map, silent_mode=silent_mode)
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low = 0,high = 255,
                                                shape = (self.game.height,self.game.width,3),
                                                dtype = np.uint8)

        self.path_length = len(self.game.path)
        self.max_step = len(self.game.available_grip) - 1 
        self.done = False 
        self.goal_reach = False
        if limit_step : 
            self.step_limit = self.max_step

        else :
            self.step_limit = 1e11

        self.reward_step_counter = 0 
        self.getting_closer_wall_step = 0
        self.dis2wall = 0  
        self.target_wall = None 
        self.close_enough = False 
    def reset(self,
              *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,):
        super().reset(seed=seed)
        self.game.reset()
        self.done = False 
        self.goal_reach = False
        self.reward_step_counter = 0 
        self.getting_closer_wall_step = 0 
        self.close_enough = False
        obs = self._generate_observation()

        return obs,{} 
    

    def _generate_observation(self):
        obs = np.zeros((self.game.height,self.game.width),dtype=np.uint8)
        obs.fill(225)
        transposed_points = np.transpose(self.game.available_grip)
        transposed_points[[0, 1]] = transposed_points[[1, 0]]
        obs[tuple(transposed_points)] = 255
        # obs[tuple(np.transpose(self.game.available_grip))] = 0.0
        transposed_points = np.transpose(self.game.path)
        transposed_points[[0, 1]] = transposed_points[[1, 0]]
        obs[tuple(transposed_points)] = np.linspace(200,50, len(self.game.path),dtype=np.uint8)
        obs = np.stack((obs, obs, obs), axis=-1)
        # obs[tuple(np.transpose(self.game.path))] = np.linspace(0.2,0.8, len(self.game.path),dtype=np.float32)
        obs[tuple((self.game.path[-1][1],self.game.path[-1][0]))] = [255,0,0] 
        transposed_points = np.transpose(self.game.wall)
        transposed_points[[0, 1]] = transposed_points[[1, 0]]
        obs[tuple(transposed_points)] = [0,0,0]
        # obs[tuple(np.transpose(self.game.wall))] = -0.5
        obs[tuple((self.game.goal[1],self.game.goal[0]))] = [0,0,255]
        # obs[tuple(self.game.goal)] = -0.8 
        # cv2.imshow("test",obs)
        # cv2.waitKey(0)
        return obs
    
    def render(self):
        self.game.render()


    def action_masks(self) -> List[bool]:
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    def get_action_mask(self,instance):
        # print(f"get dummy : {dummy}")
        return np.array([[instance._check_action_validity(a) for a in range(instance.action_space.n)]])


    def _check_action_validity(self,action):
        if len(self.game.path) <2:
            return True
        
        curr_pos = self.game.path[-1]
        x,y = self.game.path[-1]
        prev_pos = self.game.path[-2]
        if action == 0 :
            y -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            y += 1
        elif action == 4:
            x += 1
            y -= 1
        elif action == 5:
            x -= 1
            y -= 1
        elif action == 6:
            x += 1
            y += 1 
        elif action == 7:
            x -= 1
            y += 1

        next_pos = [x,y]
        prev_v = np.asarray(curr_pos) -np.asarray(prev_pos)
        next_v = np.asarray(next_pos) - np.asarray(curr_pos)

        dot = np.dot(prev_v,next_v)
        magnitude1 = np.linalg.norm(prev_v)
        magnitude2 = np.linalg.norm(next_v)
        mag = magnitude1 * magnitude2   
        if mag :
            cosine_angle = dot / mag
            angle_rad = np.arccos(cosine_angle)
            if angle_rad >= 1.6:
                return False 
            else :
                return True 
        else :
            return False


    def step(self,action):
        self.done,info = self.game.step(action)
        obs = self._generate_observation()
        reward = 0.0
        
        # Calculate distances
        cur_dis2goal = np.linalg.norm(info["path_last_pos"]-info["goal_pos"])
        prev_dis2goal = np.linalg.norm(info["path_prev_pos"]-info["goal_pos"])
        
        # Calculate wall distance if closest_wall is available
        wall_dist = 0
        if info["closest_wall"] is not None:
            wall_dist = np.linalg.norm(info["path_last_pos"]-info["closest_wall"])
        
        # Goal Reached
        if self.done and info["goal_reach"]:
            # Efficiency bonus: shorter path = higher reward
            max_plan_step = np.linalg.norm(info["start_pos"]-info["goal_pos"]) * 3
            efficiency = max(0, (max_plan_step - info["path_length"]) / max_plan_step)
            reward = 15.0 + 10.0 * efficiency
            print(f"Goal reached! Reward: {reward:.2f}")
        
        # Crash / Hit Wall
        elif self.done and not info["goal_reach"]:
            reward = -10.0
            self.target_wall = None
            self.close_enough = False
        
        # Normal step
        else:
            # 1. Left Wall Following Reward (PRIMARY OBJECTIVE)
            if info["follow_wall"]:
                reward += 2.0  # Strong reward for following left wall
                
                # 2. Wall Distance Maintenance (optimal range: 15-25 pixels)
                if 15 < wall_dist < 25:
                    reward += 0.5  # Good distance from wall (~20 pixels)
                elif wall_dist >= 30:
                    reward -= 0.5  # Too far from wall (>30 pixels)
                elif wall_dist < 10:
                    reward -= 0.3  # Too close to wall (<10 pixels)
                
                # 3. Left Turn Bonus when following wall
                # Actions: 0=Up, 1=Right, 2=Left, 3=Down, 4=UpRight, 5=UpLeft, 6=DownRight, 7=DownLeft
                # Left actions relative to forward: 0 (up), 2 (left), 5 (up-left), 7 (down-left)
                left_actions = [0, 2, 5, 7]
                if action in left_actions:
                    reward += 1.0  # Bonus for left turns around obstacles
            
            # 4. Goal Progress
            if cur_dis2goal < prev_dis2goal:
                reward += 0.3  # Getting closer to goal
            else:
                reward -= 0.3  # Moving away from goal
            
            # Reset state when starting to follow wall
            if info["follow_wall"]:
                self.target_wall = None
                self.close_enough = False
            
            # 5. Approach wall if not following
            if not info["follow_wall"] and info["closest_wall"] is not None:
                if self.target_wall is None:
                    self.target_wall = info["closest_wall"]
                
                cur_dis2wall = np.linalg.norm(info["path_last_pos"]-self.target_wall)
                if cur_dis2wall < prev_dis2goal:  # Actually moving toward wall
                    reward += 0.5
        
        return obs, reward, self.done, False ,info 
    







NUM_EPISODES = 100
RENDER_DELAY = 0.001
from matplotlib import pyplot as plt
from PIL import Image

if __name__ == "__main__":
    im = Image.open("HKSB_6F.pgm")
    im_array = np.asarray(im)
    im_down = cv2.pyrDown(im_array)
    im_down = cv2.pyrDown(im_down)
    env = PlanEnv(silent_mode=False,map=im_down)
    check_env(env)
    # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # # for i in range(NUM_EPISODES):
    # #     num_success += env.reset()
    # # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # #Action [0:Up 1:Right 2:Left 3:Down 4:Up Right 5:Up Left 6:Down Right 7:Down Left]
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #         obs,_ = env.reset()
    #         done = False
    #         i = 0
    #         while not done:
    #             plt.imshow(obs,cmap="gray")
    #             plt.show()
    #             action = env.action_space.sample()
    #             action = action_list[i]
    #             i = (i + 1) % len(action_list)
    #             obs, reward, done, _,info = env.step(action)
    #             sum_reward += reward
    #             if np.absolute(reward) > 0.001:
    #                 print(reward)
    #             env.render()
                
    #             # time.sleep(RENDER_DELAY)
    #         # print(info["snake_length"])
    #         # print(info["food_pos"])
    #         # print(obs)
    #         print("sum_reward: %f" % sum_reward)
    #         print("episode done")
    #         # time.sleep(100)
        
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))

