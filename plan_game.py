import sys
import random
import numpy as np 
import pygame
from PIL import Image
import cv2 
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance

RIGHT_TURN = -1
LEFT_TURN = 1
COLLINEAR = 0


class PlanGame:
    def __init__(self,seed = 0, map = [],silent_mode = True):
        self.silent_mode = silent_mode
        self.map = map
        self.width = map.shape[1]
        self.height = map.shape[0]
        

        if not silent_mode : 
            pygame.init()
            pygame.display.set_caption("Plan Game")
            self.screen = pygame.display.set_mode((self.width,self.height),pygame.RESIZABLE)
            self.font = pygame.font.Font(None,36)
        else : 
            self.screen = None
            self.font = None

        self.path = None
        self.follow_path = [] 
        self.wall = None
        self.wall_Q1 = None
        self.wall_Q2 = None 
        self.wall_Q3 = None
        self.wall_Q4 = None 
        self.center = None  
        self.available_grip= None 
        self.goal = None
        self.seed_value = seed
        self.direction = None 
        random.seed(seed)
        self.reset()

    def reset(self):
        # print("reset called ")
        
        self.path = [self._generate_start()]
        self.goal = self._generate_goal()
        self.wall = self._generate_wall()
        self.follow_path = []

    def _generate_start(self):
        # print("generate start called")
        a_map = np.where(self.map==255)
        a_map = list(zip(a_map[1],a_map[0]))
        start = random.sample(a_map,1)[0]
        # self.available_grip = np.delete(a_map,np.where(a_map == start))
        # print(a_map)
        # print(len(a_map))
        a_map.remove(start)
        # print(len(a_map))
        self.available_grip = a_map
        # print(f"max_step : {len(self.available_grip)}")
        # print("start")
        # print(start)
        return start 
        
    def _generate_goal(self):
        # print(self.available_grip)
        goal = random.sample(self.available_grip,1)[0]
        # print("goal : ")
        # print(goal)
        return goal

    def _generate_wall(self):
        edge = cv2.Canny(self.map,100,200)
        # print("edge shape")
        # print(edge.shape)
        vertices = np.where(edge>0)
        vertices = list(zip(vertices[1],vertices[0]))
        # print(f"wall length : {len(vertices)}")
        self._sorting_point(vertices)
        return vertices
    

    def _sorting_point(self,points):
        # points.sort(key=lambda x:[x[0],x[1]])
        points_array = np.asarray(points)
        x_median = np.median(points_array[:,0])
        y_median = np.median(points_array[:,1])
        self.center = (x_median,y_median)
        select_points = points_array[points_array[:, 0] > x_median ]
        # print(f"{len(select_points)}")
        self.wall_Q1 = select_points[select_points[:,1]< y_median]
        self.wall_Q4 = select_points[select_points[:,1]> y_median]
        select_points = points_array[points_array[:, 0] < x_median ]
        self.wall_Q3 = select_points[select_points[:,1]> y_median]
        self.wall_Q2 = select_points[select_points[:,1]< y_median]
        # return self.wall_Q2


    def _check_follow_wall(self,x,y):
        
        target = self.wall
        # if (x >= self.center[0] and y <= self.center[1]):
        #     #check Q1 
        #     target = self.wall_Q1  

        # elif(x< self.center[0] and y < self.center[1]):
        #     #check Q2 
        #    target = self.wall_Q2 
        # elif(x < self.center[0] and y > self.center[1]):
        #     #check Q3 im_array = np.asarray(im)
        #     target = self.wall_Q3
        # else :
        #     #check Q4 
        #     target = self.wall_Q4

        find = False 
        while not find :
            
            distances = distance.cdist([(x,y)],target)[0]
            if len(distances)>0:
                min_dist = np.min(distances)
                index = np.where(distances==min_dist)[0][0]
            
                closest_point = target[index]
                path_vector = np.asarray(self.path[-1]) - np.asarray(self.path[-2])
                wall_vector = closest_point - np.asarray(self.path[-2])
                cross_product = np.cross(path_vector,wall_vector)
                if cross_product > 0 :
                    find = True
                    closest_wall = closest_point
                else : 
                    
                    target = np.delete(target,index,axis=0)
            else : 
                return False,None         
        # print(distances)
        if min_dist < 7:
            # index = np.where(distances==min_dist)[0][0]
            # wall_point = target[index]
            # path_vector = np.asarray(self.path[-1]) - np.asarray(self.path[-2])
            # wall_vector = wall_point - np.asarray(self.path[-2])
            # cross_product = np.cross(path_vector,wall_vector)
            # if cross_product > 0 :
            return True,closest_wall 
        return False,closest_wall 


    def draw_path(self):
        color = (0,0,255)
        for element in self.path:
            pygame.draw.rect(self.screen,color,(element[0],element[1],3,3))

    def draw_wall(self):
        color = (0,255,0)
        for element in self.wall:
            pygame.draw.rect(self.screen,color,(element[0],element[1],2,2))


    def step(self,action):
        
        x,y = self.path[-1]
        
        #Action [0:Up 1:Right 2:Left 3:Down 4:Up Right 5:Up Left 6:Down Right 7:Down Left]
        if action != -1:
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

            self.path.append((x,y))

            if x >= self.width or y >= self.height or x<0 or y <0:
                Done = True 
                Goal_reach = False
                self.path.pop()
                info = {
                "path_length" : len(self.path),
                "follow_length" : len(self.follow_path),
                "follow_wall" : False,
                "path_last_pos" : np.array(self.path[-1]),
                "path_prev_pos" : np.array(self.path[-2]),
                "start_pos": np.array(self.path[0]),
                "goal_pos" : np.array(self.goal),
                "goal_reach" : Goal_reach
                 }
                return Done,info


            if (x,y) == self.goal:
                Goal_reach = True   
                Done = True 
                # print("Goal reached")
            else:
                Goal_reach = False 
                # print(self.path)
                
                if (x,y) in self.available_grip:
                    self.available_grip.remove((x,y))
                else :
                    Done = True

            if (x,y) in self.wall or (x,y) in self.path[:-1]:
                Done = True
            else :
                Done = False 

            following, wall_point  = self._check_follow_wall(x,y)
            # print("wall point : ")
            # print(wall_point)
            if following:
                if self.follow_path:
                    self.follow_path.append([x,y])
                    
                else :
                    self.follow_path = [[x,y]]
                    
            

            
            info = {
                "path_length" : len(self.path),
                "follow_length" : len(self.follow_path),
                "follow_wall" : following,
                "closest_wall" : wall_point,
                "path_last_pos" : np.array(self.path[-1]),
                "path_prev_pos" : np.array(self.path[-2]),
                "start_pos": np.array(self.path[0]),
                "goal_pos" : np.array(self.goal),
                "goal_reach" : Goal_reach
            }


            return Done,info
        return 0,0
    
    def draw_one_pt(self,point,color):
        pygame.draw.rect(self.screen,color,(point[0],point[1],5,5))

    def render(self):
        self.screen.fill((0,0,0))
        self.draw_wall()
        # map_reshape = self.map.reshape(self.map.shape[1],self.map.shape[0])
        # surface = pygame.surfarray.make_surface(self.map)
        # rotated_surface = pygame.transform.rotate(surface, 90)
        # self.screen.blit(surface, (0, 0))
        pygame.draw.rect(self.screen,(255,0,0),(self.goal[0],self.goal[1],5,5))
        self.draw_path()
        
        # rotated_screen = pygame.transform.rotate(self.screen,270)
        # self.screen.blit(rotated_screen, (0, 0))
        pygame.display.flip()
        


if __name__ == "__main__":
    seed = random.randint(0,1e9)
    im = Image.open("HKSB_6F.pgm")
    im_array = np.asarray(im)
    print(im_array.shape)
    x = int((im_array.shape[0]+1)/4)
    y = int((im_array.shape[1]+1)/4)
    im_down = cv2.pyrDown(im_array)
    im_down = cv2.pyrDown(im_down)
    game = PlanGame(seed=seed,map=im_down,silent_mode=False)
    print(game.width,game.height)
    pygame.init()
    game_state = 'running'
    update_interval = 0.15
    start_time = time.time()
    action = -1
    step = 0
    while True:
        for event in pygame.event.get():
            if game_state == 'running':
               
                if event.type == pygame.KEYDOWN:
                    step = step + 1
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                        action = 4
                    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                        action = 5
                    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                        action = 6 
                    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                        action = 7
                    elif keys[pygame.K_DOWN] :
                        action = 3
                    elif keys[pygame.K_UP]:
                        action = 0
                    elif keys[pygame.K_LEFT]:
                        action = 2
                    elif keys[pygame.K_RIGHT]:
                        action = 1
                
                if time.time()-start_time > update_interval:
                    
                    done,info = game.step(action)
                    if step > 1:
                        print(info["path_length"])
                        print(info["follow_length"])
                        if np.linalg.norm(info["path_last_pos"]-info["goal_pos"]) < np.linalg.norm(info["path_prev_pos"]-info["goal_pos"]):
                            print("getting closer")
                        else :
                            print("getting away")
                    game.render()
                    start_time = time.time()
                    if done :
                        game_state = "finished"

            
            if event.type == pygame.QUIT or game_state == "finished":
                pygame.quit()
                sys.exit()    

        pygame.time.wait(1)









# im = Image.open("HKSB_6F.pgm")
# im_array = np.asarray(im)
# white_image = np.ones(im_array.shape, dtype=np.uint8) * 255
# edge = cv2.Canny(im_array,100,200)
# print(edge.shape)
# vertices = np.where(edge>0)
# vertices = list(zip(vertices[0],vertices[1]))
# for point in vertices:
#     white_image[point[0]][point[1]] =0
# # contours = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # cv2.drawContours(im_array, contours, -1, (0, 0, 255), thickness=2)
# # for contour in contours:
# #     for element in contour:
# #         vertices.append(element)

# # print(len(vertices))
# # # # vertices = vertices.reverse()
# # # contours = [np.array(vertices, dtype=np.int32)]
# # # # print(contours)
# # # cv2.drawContours(im_array, contours, -1, (0, 0, 255), thickness=2)
# cv2.imshow("test",white_image)
# cv2.waitKey(0)