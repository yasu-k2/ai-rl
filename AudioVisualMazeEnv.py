import itertools
import random
import sys
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame.locals import *
from pymdp.maths import softmax
from pymdp.maths import spm_log_single as log_stable
import seaborn as sns
 
sys.setrecursionlimit(10**6)
 
random_seed = 4
direction = [(0, -2), (0, 2), (-2, 0), (2, 0)] 


def print_maze(maze_array):
    for dy_list in maze_array:
        for item in dy_list:
            if item == 0:
                print("　", end="") # Path
			elif item == 1:
				print("■", end="") # Wall
			elif item == 2:
				print("◎", end="") # Start
			elif item == 3:
				print("★", end="") # Goal	
			else:
				pass
		print("")
		
 
def fn_create_maze(updX, updY):
	rnd_array = list(range(random_seed))
	random.shuffle(rnd_array)
	
	for index in rnd_array:
		if updY + direction[index][1] < 1 or updY + direction[index][1] > maze_height-1:
			continue
		elif updX + direction[index][0] < 1 or updX + direction[index][0] > maze_width-1:
			continue
		elif maze_array[updY+direction[index][1]][updX+direction[index][0]] == 0:
			continue
		else:
			pass
			
		maze_array[updY+direction[index][1]][updX+direction[index][0]] = 0
		if index == 0:
			maze_array[updY+direction[index][1]+1][updX+direction[index][0]] = 0
		elif index == 1:
			maze_array[updY+direction[index][1]-1][updX+direction[index][0]] = 0
		elif index == 2:
			maze_array[updY+direction[index][1]][updX+direction[index][0]+1] = 0
		elif index == 3:
			maze_array[updY+direction[index][1]][updX+direction[index][0]-1] = 0
		else:
			pass
		
		#sleep(0.2)
		#print_maze(maze_array)
		fn_create_maze(updX+direction[index][0], updY+direction[index][1])
 

def create_maze(maze_width=21, maze_height=21, create=False, seed=0, 
                start_pos=(1, 1), end_pos='LowerRight',DIR='/content/drive/MyDrive/final_task'):
  """
  create = Falseならテンプレートを読み込む(21 x 21、41 x 41限定)
  end_pos='LowerRight' or 'rand'
  DIRにはテンプレートnpyを保存したディレクトリを指定
  """

  if not create:
      if ((maze_width, maze_height)==(21,21)) | ((maze_width, maze_height)==(41,41)):
          np.load(DIR+'/maze_array_h{}_w{}.npy'.format(maze_height, maze_width))
          maze_array = np.load(DIR+'/maze_array_h{}_w{}.npy'.format(maze_height, maze_width))
          maze_array[start_pos] = 0 # 初期位置設定
      else:
          create = True

  if create:
      maze_array = np.ones((maze_height, maze_width))
      maze_array[start_pos] = 0 # 初期位置設定
      np.random.seed(seed)
      print('start to create {} x {} maze'.format(maze_height, maze_width))
      fn_create_maze(start_pos[0], start_pos[1])
      maze_array[start_pos] = 2
      print("finish")
  if end_pos=='LowerRight':
      end_pos = (maze_height - 2, maze_width - 2)
  elif end_pos=='rand':
      end_ind = np.random.choice(np.where(maze_array.ravel()==0)[0])
      end_pos = np.unravel_index(end_ind, maze_array.shape)
  maze_array[end_pos] = 3 # ゴール位置設定
  # fn_print_maze(maze_array)
  return maze_array


class AudioVisualMazeEnv():
    
    def __init__(self, start_pos = (1,1), n_div_sound=10, **kwargs):

        self.maze_array = create_maze(start_pos=start_pos, **kwargs)
        self.end_pos = list(zip(*np.where(self.maze_array==3)))[0]
        self.bins = self.make_distance_bins(n_div_sound)

        self.start_pos = start_pos
        self.init_sound = self.compute_sound(self.start_pos[0], self.start_pos[1])
        self.init_state = self.start_pos + (self.init_sound,)
        self.current_state = self.init_state
        print(f'Starting state is {start_pos}')
    
    def make_distance_bins(self,n_div_sound):
        M = int(round((self.maze_array.shape[0]**2+self.maze_array.shape[1]**2)**0.5)) # 最大の距離
        return np.linspace(1, n_div_sound, M)

    def step(self,action_label):

        (Y, X, sound) = self.current_state
        if action_label == "UP":           
            Y_new = Y - 1 if Y > 0 else Y
            X_new = X

        elif action_label == "DOWN": 
            Y_new = Y + 1 if Y < 2 else Y
            X_new = X

        elif action_label == "LEFT": 
            Y_new = Y
            X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
            Y_new = Y
            X_new = X +1 if X < 2 else X

        elif action_label == "STAY":
            Y_new, X_new = Y, X 
        
        if self.maze_array[Y_new, X_new]==1: # 壁には進めない
            Y_new, X_new = Y, X
        
        sound_new = compute_sound(Y_new, X_new)
        self.current_state = (Y_new, X_new, sound_new) # store the new grid location
        obs = self.current_state # agent always directly observes the grid location they're in 
        
        # 終了判定
        if self.maze_array[Y_new, X_new]==2: # 壁には進めない
            reward = 1
            self.done = 1
        else:
            reward = 0
            self.done = 0

        #return obs
        return obs, reward, self.done, {}

    def compute_sound(self, Y, X):
        distance = (Y - self.end_pos[0])**2 + (X - self.end_pos[1])**2
        return np.digitize(distance, self.bins)

    def reset(self):
        self.current_state = self.init_state
        print(f'Re-initialized location to {self.init_state}')
        obs = self.current_state
        print(f'..and sampled observation {obs}')
        #return obs
        return obs, 0, 0, {}


if __name__ == '__main__':
    env = AudioVisualMazeEnv()
