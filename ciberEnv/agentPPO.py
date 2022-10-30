
from stable_baselines3 import PPO
import numpy as np

from croblink import *

SIM_IP = "127.0.0.1"
SIM_PORT = 6000

model = PPO.load("ciber_ppo_1")

rob = CRobLink("agentPPO", 0, 'localhost')

while True:
    rob.readSensors()

    obsl = [int(x) for x in rob.measures.lineSensor]
    obs = np.array(obsl)

    action, _states = model.predict(obs, deterministic=True)

    rob.driveMotors(action[0], action[1])

