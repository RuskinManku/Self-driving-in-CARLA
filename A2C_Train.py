from torch.nn.modules.linear import Linear
from torch.nn.modules.activation import ReLU
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from CarlaEnv import CarEnv
import random
import cv2
import collections
from torch.distributions.normal import Normal
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import StepLR
print(device)
from matplotlib import pyplot
class AutoEncoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super(AutoEncoder, self).__init__()

        ### Convolutional section
        num_input_channels = 3
        base_channel_size = 16
        latent_dim =encoded_space_dim
        act_fn=nn.GELU
        c_hid = base_channel_size
        self.lin_out=2048
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2), # 64x64 => 32x32
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Conv2d(4*c_hid, c_hid, kernel_size=3, padding=1, stride=2), # 4x4 => 2x2
            act_fn(),
            nn.Conv2d(c_hid, latent_dim, kernel_size=3, padding=1, stride=2), # 2x2 => 1x1
            act_fn(),
            nn.Flatten()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 1x1 => 2x2
            act_fn(),
            nn.ConvTranspose2d(c_hid, 4*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 2x2 => 4x4
            act_fn(),
            nn.ConvTranspose2d(4*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 32x32 => 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        x=x.permute(2,0,1).unsqueeze(0)
        input=x.clone().detach()
        x = self.encoder_cnn(x)
        intermediate=x.clone().detach()
        x = x.reshape(x.shape[0], -1, 1, 1)
        x = self.decoder_conv(x)
        return x,intermediate,input

class PolicyNetwork(nn.Module):
    """
    Implementation of NN to predict the actor gaussian policy for both REINFORCE and A2C.
    """
    def __init__(self,observation_space_size,action_space_size):
        super(PolicyNetwork, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(observation_space_size,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,action_space_size)
        )
    def forward(self,state):
        state=torch.Tensor(state).to(device)
        return self.model(state).squeeze(0)

class ValueNetwork(nn.Module):
    """
    Implementation of NN to predict the value function for A2C.
    """
    def __init__(self,observation_space_size):
        super(ValueNetwork, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(observation_space_size,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self,state):
        state=torch.Tensor(state).to(device).unsqueeze(0)
        return self.model(state).squeeze(0)

class A2C():
    """
    Implementation of Online Actor-Critic Algorithm
    """
    def __init__(self,env,hidden_size,action_space_size):
        self.autoencoder=AutoEncoder(encoded_space_dim=hidden_size).to(device)
        self.action_space_size=action_space_size
        self.policy_network=PolicyNetwork(observation_space_size=hidden_size,action_space_size=2*action_space_size).to(device)
        self.value_network=ValueNetwork(observation_space_size=hidden_size).to(device)
        self.env=env
    def run_episodes(self,num_episodes,std_dev=0.5,lr_actor=1e-3,lr_critic=1e-4,lr_autoencoder=1e-6,reward_log_stepsize=100,buffer_size=5,image_size=(480,480,3),gamma=0.99,switch_train_steps=20):
        """
        Primary function for running num_episodes episodes of trajectories.
        """
        self.autoencoder.load_state_dict(torch.load("ae.pth"))
        # self.policy_network.load_state_dict(torch.load("policy.pth"))
        self.autoencoder.train()
        self.policy_network.train()
        self.value_network.train()
        self.optimizer_actor=torch.optim.Adam(self.policy_network.parameters(),lr=lr_actor)
        self.optimizer_critic=torch.optim.Adam(self.value_network.parameters(),lr=lr_critic)
        self.optimizer_autoencoder=torch.optim.Adam(self.autoencoder.parameters(),lr=lr_autoencoder)
        scheduler = StepLR(self.optimizer_autoencoder, step_size=1, gamma=0.1)
        self.criterion_ae=nn.MSELoss(reduction='mean')
        cum_reward_stepsize=0
        avg_rewards=[]
        mean_ae_loss=0
        ae_steps=0
        TRAIN_AE=True
        TRAIN_A2C=False
        for episode in range(num_episodes):
            observation,_=self.env.reset()
            img_observation=observation
            observation=torch.Tensor(observation).to(device)
            decoded_observation,state,input_observation=self.autoencoder(observation)
            curr_grad_step=0
            while(1):
                #MSE loss step for autoencoder
                loss_ae=self.criterion_ae(input_observation,decoded_observation)
                inp_show=np.uint8(input_observation.clone().detach().cpu().squeeze(0).permute(1,2,0)*255)
                out_show=np.uint8(decoded_observation.clone().detach().cpu().squeeze(0).permute(1,2,0)*255)
                filename=f"Image/{curr_grad_step}_in.jpg"
                # filename=f"Image/Images/{episode}_{curr_grad_step}.jpg"
                pyplot.imsave(filename, inp_show)
                filename=f"Image/{curr_grad_step}_out.jpg"
                pyplot.imsave(filename, out_show)
                print(loss_ae)
                assert 1==0
                mean_ae_loss+=loss_ae
                ae_steps+=1
                if TRAIN_AE:
                    self.optimizer_autoencoder.zero_grad()
                    loss_ae.backward()
                    self.optimizer_autoencoder.step()
                # Calculate the mean and std dev of gaussian policy
                action_stats=self.policy_network(state)
                action_means=torch.tanh(action_stats[:self.action_space_size])
                action_stddevs=torch.relu(action_stats[self.action_space_size:])+2e-1 #sigmoid so our std dev is between 0 and 1 and not negative or too large
                # Create a distribution from that mean and passed std_dev
                action_dist=Normal(loc=action_means,scale=action_stddevs)
                # Sample an action from the gaussian policy
                action=action_dist.sample()
                curr_grad_step+=1
                # Calculate log probability of that action
                action_logprob=action_dist.log_prob(action)
                action=action.detach().cpu().numpy()
                # filename=f"Images/{curr_grad_step}_{action}.jpg"
                # pyplot.imsave(filename, img_observation)
                next_observation, reward, terminated, info = self.env.step(action)
                img_observation=next_observation
                next_observation=torch.Tensor(next_observation).to(device)
                next_decoded_observation,next_state,next_input_observation=self.autoencoder(next_observation)
                cum_reward_stepsize+=reward
                # The target for the critic, reward+V(next_state), and if our state is terminal, then it's only the reward
                critic_target=(reward+(1-terminated)*gamma*self.value_network(next_state)).detach()
                # The advantage function= critic_target-V(state)
                advantage=critic_target-self.value_network(state)

                if TRAIN_A2C:
                    # Critic Step
                    self.optimizer_critic.zero_grad()
                    critic_loss=torch.pow(advantage,2) #MSE for critic_loss
                    critic_loss.backward()
                    self.optimizer_critic.step()
                    #Actor step
                    self.optimizer_actor.zero_grad()
                    actor_loss=-action_logprob*(advantage.detach()) #Loss=-1*action_logprob*advantage
                    actor_loss.sum().backward()
                    self.optimizer_actor.step()

                state=next_state
                observation=next_observation
                decoded_observation=next_decoded_observation
                input_observation=next_input_observation
                if terminated:
                    break

            #True False Changing
            if (episode+1)==200:
                scheduler.step()
                TRAIN_A2C=not TRAIN_A2C
                print(f"Switching, now, TRAIN_AE:{TRAIN_AE}, TRAIN_A2C:{TRAIN_A2C}")
            #Logging
            if (episode+1)%reward_log_stepsize==0:
                avg_rewards.append(cum_reward_stepsize/reward_log_stepsize)
                mean_ae_loss=mean_ae_loss/ae_steps
                print(f"Episode:{episode+1}, Average reward of last {reward_log_stepsize} steps: {cum_reward_stepsize/reward_log_stepsize}, MSE:{mean_ae_loss}")
                mean_ae_loss=0
                ae_steps=0
                cum_reward_stepsize=0
        return self.policy_network,avg_rewards


def main():
    env=CarEnv()
    im_shape=(3,env.im_height,env.im_width)
    action_space_size=2
    hidden_size=400
    a2c=A2C(env=env,hidden_size=hidden_size,action_space_size=action_space_size)
    policy_2,reward_history_2=a2c.run_episodes(num_episodes=100,lr_actor=1e-3,lr_critic=1e-3,lr_autoencoder=5e-5,reward_log_stepsize=1,gamma=0.85)
    # torch.save(reward_history_2, "reward_history_AE.pth")
    torch.save(a2c.autoencoder.state_dict(), "ae.pth")

if __name__=='__main__':
    main()
