import gymnasium as gym 

env_name = "CartPole-v1"
env = gym.make(env_name,render_mode="human")
observation,info = env.reset()

print("Starting Obseravtion :",observation)
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]


episode_over = False
total_reward = 0


while not episode_over:

    #step 1 : choose an action , 0 = push cart left 1= push cart right 
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")

env.close()
