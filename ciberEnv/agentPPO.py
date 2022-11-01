
from stable_baselines3 import PPO
import numpy as np

import sys
sys.path.append('../pClient')
from croblink import *

SIM_IP = "127.0.0.1"
SIM_PORT = 6000

model = PPO.load("ciber_ppo_3")

rob = CRobLink("agentPPO", 0, 'localhost')

action = np.array([0.0,0.0])
while True:
    rob.readSensors()

    obsl = [float(x) for x in rob.measures.lineSensor]
    obs = np.append(np.array(obsl),action)

    action, _states = model.predict(obs, deterministic=True)

    rob.driveMotors(action[0], action[1])

