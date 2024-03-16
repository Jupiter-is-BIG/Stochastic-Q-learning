import gym
from IPython import display
import matplotlib.pyplot as plt
import time
x = [
    "SFFFFFFFH",
    "FFHHHHHFH",
    "FFHFFFFFH",
    "FFHFFHFFH",
    "FFHFFHFFH",
    "FFFFFHFFG",
]

env = gym.make('FrozenLake-v1', render_mode='rgb_array', desc=x, is_slippery=False) # insert your favorite environment
print(env.action_space)
env.reset()
img = plt.imshow(env.render()) # only call this once
for _ in range(10):
    img.set_data(env.render()) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = 1
    env.step(action)
    time.sleep(0.1)