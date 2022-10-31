from gym import Env,spaces
import numpy as np
import subprocess
import socket
import time
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

import sys
sys.path.append('../pClient')
import croblink

SIM_IP = "127.0.0.1"
SIM_PORT = 6000

class CiberEnv(Env):
    def __init__(self) -> None:
        super().__init__()

        #self.observation_shape=(7,)

        #self.observation_space=spaces.MultiBinary(7)

        self.observation_space=spaces.Box(low=np.array([0,0,0,0,0,0,0,-0.15,-0.15]), 
                                         high=np.array([1,1,1,1,1,1,1,0.15,0.15]),
                                         shape=(9,),dtype=np.float32)

        self.action_space=spaces.Box(low=-0.15, high=0.15,shape=(2,),dtype=np.float32)

        self.sim_proc = subprocess.Popen(['../simulator/simulator', "--param","../Labs/rmi-2223/C1-env-config.xml",
                          "--lab","../Labs/rmi-2223/C1-lab.xml", "--grid", "../Labs/rmi-2223/C1-grid.xml",
                          "--scoring","4"])

        time.sleep(5)

        self.prev_score = 0

        #self.ctlsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #self.ctlsock.sendto('<Viewer/>', (SIM_IP, SIM_PORT))

        #self.agentapi = croblink.CRobLink('ciberEnv',1,SIM_IP)

        
    def step(self, action):
        self.agentapi.driveMotors(action[0],action[1])
        self.agentapi.readSensors()

        #obsl = [int(x) for x in self.agentapi.measures.lineSensor]
        obsl = [float(x) for x in self.agentapi.measures.lineSensor]
        obs = np.append(np.array(obsl),action)

        done = self.agentapi.measures.time == 5000

        reward = self.agentapi.measures.score - self.prev_score
        self.prev_score = self.agentapi.measures.score

        if done:
            print("SCORE ENV", self.agentapi.measures.score)

        return obs, reward, done, {"score":self.agentapi.measures.score}


    def reset(self):
        if(hasattr(self,"agentapi") ):
            self.agentapi.reset()
        else: 
            self.agentapi = croblink.CRobLink('ciberEnv',1,SIM_IP)
        self.agentapi.readSensors()
        #obsl = [int(x) for x in self.agentapi.measures.lineSensor]
        obsl = [float(x) for x in self.agentapi.measures.lineSensor]
        obs = np.append(np.array(obsl),np.array([0.0,0.0]))

        return obs

    def close(self):
        self.sim_proc.terminate()



c_env = CiberEnv()

#c_env = DummyVecEnv([lambda: c_env])


#model = PPO("MlpPolicy", c_env, verbose=1)
model = PPO.load("ciber_ppo_2", env=c_env)
model.learn(1000000)


model.save("ciber_ppo_3")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("ciber_ppo_2", env=c_env)

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#print("evaluate mean", mean_reward, "std", std_reward)

while True:
    print('Testing')
    obs = c_env.reset()
    print('obs',obs,'type',type(obs),'dtype',obs.dtype)
    done = False
    while not done:
        #action = c_env.action_space.sample()
        #action = np.array([obs[0]*0.15,obs[1]*0.15])
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = c_env.step(action)
        if done:
            print('Score', info['score'])

