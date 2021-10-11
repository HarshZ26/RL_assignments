import gym
import numpy as np
import gym_minigrid.envs
from gym_minigrid.wrappers import *
import matplotlib
import matplotlib.pyplot as plt
import time

name = 'MiniGrid-FourRooms-v0'
#for epsilon=0 greedy policy uncomment and load the script file below.
#sar_val = np.load('sar_val.npy')
sar_val = np.zeros((15,15,4,3))

def policy(x,y,dr,epsilon):
	randnum = np.random.rand()
	
	if randnum < epsilon:
		return np.random.randint(0,3)
	else:
		return np.argmax(sar_val[x][y][dr])
		
def gen_episode (env,alpha,gamma,epsilon):
	good = "undone"
	env.reset()
	done = False
	#env.render()
	steps = 0
	x = env.agent_pos[0]
	y = env.agent_pos[1]
	dr = env.agent_dir
	
	while not done:
		action = policy(x,y,dr,epsilon)
		_,rew,done, _ = env.step(action)
		steps += 1
		#env.render()
		x_new = env.agent_pos[0]
		y_new = env.agent_pos[1]
		dr_new = env.agent_dir
		sar_val[x,y,dr,action] += alpha*(rew + gamma*(sar_val[x_new,y_new,dr_new,policy(x_new,y_new,dr_new,epsilon)]) - sar_val[x,y,dr,action])
		x = x_new
		y = y_new
		dr = dr_new
	if rew >0:
		good = "Done" 
	return rew,steps,good

rew_list = []
step_list = []
env = gym.make(name)
alpha = 0.5
gamma = 0.5
ep = 5000
for i in range(0,ep):
	rew,steps,ss = gen_episode(env,alpha,gamma,(500/(500+i)))
	#rew,steps,ss = gen_episode(env,alpha,gamma,0)
	print("Episode "+str(i+1)+" completed",ss)
	rew_list.append(rew)
	step_list.append(steps)
np.save('sar_val.npy',sar_val)

plt.clf()
plt.plot(rew_list)
plt.savefig('sar_rew.png')
plt.clf()
plt.plot(step_list)
plt.savefig('sar_steps.png')
np.save('sar_rewards',np.array(rew_list))
np.save('sar_steps',np.array(step_list))




