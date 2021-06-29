# Rotary-Pendulum-with-Reinforcement-learning-in-Gazebo-without-openai-gym
Reinforcement Learning is implemented on Rotary Pendulum for swing up and balancing. Proximal Policy Optimization(PPO) with continious action is used perform task in Gazebo simulation. <br><br>
<br>
<img src = "https://github.com/Keshav2829/Rotary-Pendulum-with-Reinforcement-learning-in-Gazebo-without-openai-gym/blob/main/Rotary_pendulum.png" height=250 width=500>
# Table of Contents
- [Getting Started](#getting-started)
- [Code Overview](#code-overview)
- [File description](#file-description)
<br>

## Getting Started
To getting started basic knowledge of Reinforcement Learning is required for that a Course on Reinforcement Learning offered by Univercity of Alberta can be helpful to understand basic concepts. To get the idea about PPO algorithm with implementation [this](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) can be helpful.


## Code Overview
A solid work model of Rotary Pendulum is imported into Gazebo Environment.Pytorch is used to construct Neural Network.code runs until agent takes 2000000 steps. I used batch size of 4000, so after every 4000 steps agent start learning. total 10 iterations are performed on sigle batch.To communicate with Gazebo environment Rosserial is used. A delay of 0.02 sec is used to get observation after taking an action.following sudo code is used for coding
<img src = "https://github.com/Keshav2829/Rotary-Pendulum-with-Reinforcement-learning-in-Gazebo-without-openai-gym/blob/main/sudo_code_ppo.png" height=500 width=1000>

## File Description
1. <b>actor_model.pt</b> : saved actor network.
2. <b>critic_model.pt</b> : saved critic network.
3. <b>pen.urdf</b> : solid work model exported as urdf.
4. <b>PPO_torch_actor_critic.py</b> : main file that contains PPO algorithm and gazebo environment
5. <b>gazebo.launch</b> : file that summerises all componants of gazebo environment and lauches it in gazebo environment
## Authors
* **Keshav Agrawal** - [GitHub](https://github.com/Keshav2829) | [LinkedIn](https://www.linkedin.com/in/keshav-agrawal/)
