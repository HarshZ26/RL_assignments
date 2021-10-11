import gym
import numpy as np
import gym_minigrid.envs
from gym_minigrid.wrappers import *
import matplotlib
import matplotlib.pyplot as plt
import time

name = 'MiniGrid-FourRooms-v0'

q_val = np.zeros((15,15,4,3))
#uncomment below to run for epsilon =0 policy
#q_val = np.load('q_val.npy')

def policy(x,y,dr,epsilon):
	randnum = np.random.rand()
	
	if randnum < epsilon:
		return np.random.randint(0,3)
	else:
		return np.argmax(q_val[x][y][dr])
		
def gen_episode (env,alpha,gamma,epsilon):
	good = "undone"
	env.reset()
	done = False
	steps = 0
	x = env.agent_pos[0]
	y = env.agent_pos[1]
	dr = env.agent_dir
	
	while not done:
		action = policy(x,y,dr,epsilon)
		_,rew,done, _ = env.step(action)
		steps += 1
		env.render()
		x_new = env.agent_pos[0]
		y_new = env.agent_pos[1]
		dr_new = env.agent_dir
		q_val[x,y,dr,action] += alpha*(rew + gamma*(np.max(q_val[x_new,y_new,dr_new])) - q_val[x,y,dr,action])
		x = x_new
		y = y_new
		dr = dr_new
	if rew >0:
		good ="Done"
	return rew,steps,good


rew_list = []
step_list = []
env = gym.make(name)
alpha = 0.4
gamma = 0.6
ep = 500
for i in range(0,ep):
	rew,steps,ss = gen_episode(env,alpha,gamma,(50/(50+i)))
	#rew,steps,ss = gen_episode(env,alpha,gamma,0)
	print("Episode "+str(i+1)+" completed,",ss)
	rew_list.append(rew)
	step_list.append(steps)
np.save('q_val.npy',q_val)

plt.clf()
plt.plot(rew_list)
plt.savefig('qrew.png')
plt.clf()
plt.plot(step_list)
plt.savefig('q_steps.png')
np.save('q_rewards',np.array(rew_list))
np.save('q_steps',np.array(step_list))

