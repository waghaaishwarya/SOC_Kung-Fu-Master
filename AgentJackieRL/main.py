import gymnasium as gym
import numpy as np
import cv2
import os
from collections import deque
from agent import DQNAgent
from memory import ReplayMemory
import ale_py
#import shimmy 

# hypeparameters 
EPISODES = 20
BATCH_SIZE = 32
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 10

# preprocessing
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([frame] *4, maxlen=4)
    else:
        stacked_frames.append(frame)

    return np.stack(stacked_frames, axis=-1), stacked_frames


# main 
env = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array")
action_size = env.action_space.n
agent = DQNAgent((84, 84, 4), action_size)
memory = ReplayMemory(MEMORY_SIZE)

for episode in range(EPISODES):
    state, _ = env.reset()
    stacked_frames = deque([np.zeros((84, 84))]* 4, maxlen=4)
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    total_reward = 0

    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_proc, stacked_frames = stack_frames(stacked_frames, next_state, False)
        memory.push(state, action, reward, next_state_proc, done)


        state = next_state_proc
        total_reward += reward

        if len(memory) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            agent.train(batch, BATCH_SIZE)

    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_model()

    print(f"episode {episode + 1}/ {EPISODES}, total reward: {total_reward}, epsilon: {agent.epsilon: .3f}")

print("Training complete!")
agent.model.save("models/jackie_model.keras")
print("Model saved as jackie_model.keras")

env.close()