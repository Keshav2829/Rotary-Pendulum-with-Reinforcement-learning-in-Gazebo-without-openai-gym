import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import matplotlib.pyplot as plt
import rospy
from rospy.names import resolve_name
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
import time

class PPOMemory:
    def __init__(self):
        self.states =[]
        self.actions = []
        self.rewards= []
        self.log_probs = []
        self.rtgs = []
    
    def store_memory(self, state, action, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
    
    def store_ep_reward(self, ep_reward):
        self.rewards.append(ep_reward)

    def clear_memory(self):
        self.states =[]
        self.actions = []
        self.rewards= []
        self.log_probs = []
        self.rtgs = []
    
    def get_observations(self):
        return self.states,self.actions, self.rewards, self.log_probs


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, layer1_dims, layer2_dims):
        super(FeedForwardNetwork,self).__init__()

        self.layer1 = nn.Linear(input_dims, layer1_dims)
        self.layer2 = nn.Linear(layer1_dims,layer2_dims)
        self.output = nn.Linear(layer2_dims,output_dims)
    
    def forward(self,state):
        if isinstance(state, np.ndarray):
            state = T.tensor(state, dtype=T.float)
        
        layer1 = F.relu(self.layer1(state))
        layer2 = F.relu(self.layer2(layer1))
        out = self.output(layer2)

        return out


class PPOAgent:
    def __init__(self, input_dim, n_action):
        self.input_dim = input_dim
        self.n_actions = n_action
        self.gamma =0.99
        self.n_iter = 10
        self.clip =0.2
        self.lr = 0.0001
        
        self.actor_path = 'actor_model.pt'
        self.critic_path = 'critic_model.pt'

        self.memory = PPOMemory()
        self.actor = FeedForwardNetwork(self.input_dim,self.n_actions,layer1_dims=256,layer2_dims=256)
        self.critic = FeedForwardNetwork(self.input_dim, 1, layer1_dims=256, layer2_dims=256)
        
        self.actor_optim = Adam(self.actor.parameters(), lr= self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr= self.lr)

        self.cov_var = T.full(size=(self.n_actions,), fill_value=0.01)
        self.cov_mat = T.diag(self.cov_var)
    
    def evaluate(self, batch_obs, batch_acts):
        v = self.critic(batch_obs).squeeze()
        mu = self.actor(batch_obs)
        std = self.cov_mat
        dist = MultivariateNormal(mu, std)
        log_prob = dist.log_prob(batch_acts)

        return v, log_prob

    def choose_action(self, state):
        mu = self.actor(state)
        dist = MultivariateNormal(mu, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob

    def learn(self):
        states, actions, rewards, old_log_probs = self.memory.get_observations()
        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        old_log_probs = T.tensor(old_log_probs, dtype=T.float)
        rtgs = self.compute_rtgs(rewards)

        old_value,_ = self.evaluate(states,actions)
        advantages = rtgs- old_value.detach()
        advantages = (advantages- advantages.mean())/(advantages.std()+ 1e-10)
        
        for _ in range(self.n_iter):
            current_v, current_log_probs= self.evaluate(states, actions)

            ratios = T.exp(current_log_probs- old_log_probs)
            surr1 = ratios*advantages
            surr2= T.clamp(ratios, 1-self.clip, 1+self.clip)*advantages
      
            actor_loss = (-T.min(surr1,surr2)).mean()
            critic_loss = nn.MSELoss()(current_v,rtgs)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)          
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self.memory.clear_memory()


    
    def compute_rtgs(self,batch_rewards):
        rtgs =[]
        for ep_rw in batch_rewards:
            for rw in range(len(ep_rw)):
                discount =1
                r=0
                for k in range(rw, len(ep_rw)):
                    r += discount*ep_rw[k]
                    discount*= self.gamma
                rtgs.append(r)
        rtgs = T.tensor(rtgs,dtype=T.float)
        return rtgs

    def remember(self, state, action, log_prob):
        self.memory.store_memory(state,action,log_prob)
    
    def remember_reward(self, ep_reward):
        self.memory.store_ep_reward(ep_reward)
    
    def save_models(self):
        print('......saving models.........')
        T.save(self.actor.state_dict(),self.actor_path)
        T.save(self.critic.state_dict(),self.critic_path)
    
    def load_models(self):
        print('.....loading models.......')
        self.actor.load_state_dict(T.load(self.actor_path))
        self.critic.load_state_dict(T.load(self.critic_path))


class GazeboEnv:
    def __init__(self,max_steps=1000):
        self.max_steps = max_steps
        rospy.init_node('talker', anonymous = True)
        self.pub = rospy.Publisher('/ip/joint_controller/command', Float64, queue_size = 10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.data = None
        self.control =0
        rospy.Subscriber('/ip/joint_states', JointState, self.callback)
    
    def callback(self,data):
        if self.control==1:
            self.data = data
            self.control = 0

    def step(self,action,n_steps):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except:
            pass
        action = np.clip(action,-1.5,1.5)[0]
        self.pub.publish(action)
        self.control=1                  ##control flag make sures that data is updated only once after taking action

        #if not load_checkpoint:
        #    agent.learn()
        time.sleep(0.02)


        
        if(self.data==None):
            while self.data is None:
                try:
                    self.data = rospy.wait_for_message('/ip/joint_states', JointState, timeout=5)
                except:
                    pass

        ##wait untill data is updated
        while(self.control==1):            
            pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except:
            pass
        
        state_= self.state_normalizaton()       
        state = np.array([np.cos(state_[0]), np.sin(state_[0]),np.cos(state_[1]),np.sin(state_[1]),state_[2],state_[3]])
        reward,done = self.get_reward_done(state_,n_steps,action)
        return state,reward,done
    
    def state_normalizaton(self):
        data_received_1 = np.concatenate((self.data.position,self.data.velocity))
        data_received_1[0] = (((data_received_1[0]+np.pi)%(2*np.pi))-np.pi)
        data_received_1[1] = (((data_received_1[1]+np.pi)%(2*np.pi))-np.pi)
        return data_received_1
    
    def get_reward_done(self,state,n_steps,action):
        joint1_position = np.absolute(state[0])-np.pi
        joint2_position = np.absolute(state[1])-np.pi
        joint1_velocity = state[2]
        joint2_velocity = state[3]
        done = False
        reward_ = -0.1*(10*(joint2_position**2)+0.005*joint2_velocity**2+0.001*action**2)
        if ((np.absolute(joint2_position)<0.17) and np.absolute(joint2_velocity)<4):
            reward_= reward_+25-((25/0.17)*np.absolute(joint2_position))

        if(n_steps>self.max_steps):
            done= True

        return reward_,done
    
    def env_reset(self):
        #self.reset_proxy()
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except:
            pass
        self.pub.publish(0)
        self.control=1
        time.sleep(20)

        while self.data is None:
            try:
                self.data = rospy.wait_for_message('/ip/joint_states', JointState, timeout=5)
            except:
                pass

        while(self.control==1):
            pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except:
            pass
        state_= self.state_normalizaton()
        state = np.array([np.cos(state_[0]), np.sin(state_[0]),np.cos(state_[1]),np.sin(state_[1]),state_[2],state_[3]])
        return state
        





if __name__=='__main__':
    env = GazeboEnv(max_steps=800)
    agent = PPOAgent(6,1)
    max_time_steps= 2000000
    batch_size = 4000
    t=0
    best_score= -3000
    transition =0

    load_model_flag= True
    training_flag = False
    
    total_learning =0 
    score_history = []
    score_history_avg = []
#load models for training or testing
    if load_model_flag is True:
        agent.load_models()

    
    while t< max_time_steps:     ##max time steps for learning 
        e_t=0

        while e_t < batch_size:     ##time steps store for traingin in a batch
            observation = env.env_reset()
            done = False
            ep_rwd=[]
            score=0
            n_steps=0
            while not done:
                #env.render()
                action, log_prob = agent.choose_action(observation)
                observation_, reward, done= env.step(action,n_steps)
                t+=1
                e_t+=1
                score+=reward
                ep_rwd.append(reward)
                agent.remember(observation,action,log_prob)
                observation=observation_
                n_steps+=1
            agent.remember_reward(ep_rwd)     #store episodic rewards

        #store values for debugging 
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            score_history_avg.append(avg_score)

            print('time_steps={}   learning_steps ={}   score={}   avg_score={}'.format(t,total_learning,score,avg_score))
        
       #start learining for a batch
       #if we want to train our model or not
        if training_flag is True:
            agent.learn()
            total_learning+=1
        
        if ((avg_score>best_score) and (training_flag is True)):
            best_score= avg_score
            agent.save_models()
       
    
    plt.plot(score_history_avg)
    plt.show()
