import gym
import numpy as np
import cv2
from collections import deque
from agent import DQNAgent


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def stack_frames(stacked_frames, frame, is_new_episode):
    processed_frame = preprocess_frame(frame)
    if is_new_episode:
        stacked_frames = deque([processed_frame]*4, maxlen=4)
    else:
        stacked_frames.append(processed_frame)
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


def train_dqn(episodes=1000):
    env = gym.make("KungFuMasterDeterministic-v4")
    action_space = env.action_space.n
    state_shape = (84, 84, 4)
    agent = DQNAgent(state_shape=state_shape, action_space=action_space)
    stacked_frames = deque([np.zeros((84, 84))]*4, maxlen=4)
    for episode in range(episodes):
        state, _ = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        total_reward = 0
        done = False
        step_count = 0
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            if step_count % 4 ==0:
                agent.replay()
            step_count += 1
            state = next_state
            total_reward += reward
        agent.update_target_network()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        print(
            f"Episode {episode+1} - Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        if (episode+1) % 50 == 0:
           agent.model.save(f"model_episode_{episode+1}.h5")
    env.close()


if __name__ == "__main__":
    train_dqn(episodes=1000)
