import os
import torch
import glob
import json
from env import ClashRoyaleEnv
from dqn_agent import DQNAgent
from datetime import datetime
from logger import Logger

def get_latest_model_path(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
    if not model_files:
        return None
    model_files.sort()  # Lexicographical sort works for timestamps
    return model_files[-1]

def train():
    logger = Logger(name="train")
    env = ClashRoyaleEnv()
    agent = DQNAgent(env.state_size, env.action_size)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Load latest model if available
    latest_model = get_latest_model_path("models")
    if latest_model:
        agent.load(os.path.basename(latest_model))
        # Load epsilon
        meta_path = latest_model.replace("model_", "meta_").replace(".pth", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                agent.epsilon = meta.get("epsilon", 1.0)
            logger.info(f"Epsilon loaded: {agent.epsilon}")

    episodes = 10000
    batch_size = 32

    for ep in range(episodes):

        state = env.reset()
        logger.info(f"Episode {ep + 1} starting. Epsilon: {agent.epsilon:.3f}")  # <-- Add this line
        total_reward = 0
        done = False
        while not done:
            action_mask = env.get_valid_action_mask(state)
            action = agent.act(state, action_mask)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            total_reward += reward
            logger.extra_visibility(f"Step: {state}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
        logger.success(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

        if ep % 10 == 0:
            agent.update_target_model()
            # Save model and epsilon every 10 episodes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join("models", f"model_{timestamp}.pth")
            torch.save(agent.model.state_dict(), model_path)
            with open(os.path.join("models", f"meta_{timestamp}.json"), "w") as f:
                json.dump({"epsilon": agent.epsilon}, f)
            logger.success(f"Model and epsilon saved to {model_path}")

if __name__ == "__main__":
    train()