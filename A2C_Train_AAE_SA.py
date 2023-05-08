from torch.nn.modules.linear import Linear
from torch.nn.modules.activation import ReLU
import torchvision.models as models
import torchvision.transforms as transforms
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
class Encoder(nn.Module):
    def __init__(self, nc, nf, nz):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, nf, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=2*4*4*nf*4, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mu = nn.Linear(in_features=1024, out_features=nz)
        #self.var = nn.Linear(in_features=1024, out_features=nz)

    def forward(self, input):
        y =  self.net(input)
        return self.mu(y)#, self.var(y)

class Unflatten(nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(len(input), self.shape[0], self.shape[1], self.shape[2])

class Decoder(nn.Module):
    def __init__(self, nc, nz, nf):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=nz, out_features=2*4*4*nf*4),
            nn.BatchNorm1d(num_features=2*4*4*nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            Unflatten((256, 4, 4)),

            # add output_padding=1 to ConvTranspose2d to reconstruct original size
            nn.ConvTranspose2d(nf*8, nf*4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # add output_padding=1 to ConvTranspose2d to reconstruct original size
            nn.ConvTranspose2d(nf*4, nf*2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(nf*2, nf, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(nf, int(nf/2), 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(int(nf/2)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int(nf/2), nc, 5, 1, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.net(input)

class AutoEncoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(nc=3, nf=32, nz=encoded_space_dim).to(device)
        self.dec = Decoder(nc=3, nz=encoded_space_dim, nf=32).to(device)
        self.transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def forward(self, x):
        x=x.permute(2,0,1).unsqueeze(0)
        x=self.transform(x)
        input=x.clone().detach()
        x = self.enc(x)
        intermediate=x.clone().detach()
        x = self.dec(x)
        return (x+1.0)/2.0,intermediate,(input+1.0)/2.0

class PolicyNetwork(nn.Module):
    """
    Implementation of NN to predict the actor gaussian policy for both REINFORCE and A2C.
    """
    def __init__(self,observation_space_size,action_space_size):
        super(PolicyNetwork, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(observation_space_size,64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,16),
            nn.LeakyReLU(),
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
            nn.Linear(observation_space_size,64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,16),
            nn.LeakyReLU(),
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
        self.autoencoder=AutoEncoder(encoded_space_dim=hidden_size)
        self.action_space_size=action_space_size
        self.policy_network=PolicyNetwork(observation_space_size=hidden_size+11,action_space_size=action_space_size).to(device)
        self.value_network=ValueNetwork(observation_space_size=hidden_size+11).to(device)
        self.env=env
    def run_episodes(self,num_episodes,std_dev=0.5,lr_actor=1e-3,lr_critic=1e-4,lr_autoencoder=1e-6,reward_log_stepsize=100,buffer_size=5,image_size=(480,480,3),gamma=0.99,switch_train_steps=20):
        """
        Primary function for running num_episodes episodes of trajectories.
        """
        self.autoencoder.enc.load_state_dict(torch.load("enc.pth"))
        self.autoencoder.dec.load_state_dict(torch.load("dec.pth"))
        self.autoencoder.eval()
        # self.autoencoder.load_state_dict(torch.load("ae.pth"))
        # self.policy_network.load_state_dict(torch.load("policy_best.pth"))
        # self.autoencoder.train()
        self.policy_network.train()
        # self.policy_network.load_state_dict(torch.load("policy.pth"))
        self.value_network.train()
        self.optimizer_actor=torch.optim.Adam(self.policy_network.parameters(),lr=lr_actor)
        self.optimizer_critic=torch.optim.Adam(self.value_network.parameters(),lr=lr_critic)
        self.optimizer_autoencoder=torch.optim.Adam(self.autoencoder.parameters(),lr=lr_autoencoder)
        # scheduler = StepLR(self.optimizer_autoencoder, step_size=1, gamma=0.1)
        self.criterion_ae=nn.MSELoss(reduction='mean')
        cum_reward_stepsize=0
        avg_rewards=[]
        avg_critics=[]
        mean_ae_loss=0
        ae_steps=0
        mean_critic_loss=0
        critic_steps=0
        mean_actor_loss=0
        actor_steps=0
        TRAIN_AE=True
        TRAIN_A2C=True
        max_ep_reward=0
        self.autoencoder_batch=[]
        BATCH_SIZE=64
        for episode in range(num_episodes):
            observation,info=self.env.reset()
            img_observation=observation
            observation=torch.Tensor(observation).to(device)
            # self.autoencoder_batch.append(observation)
            decoded_observation,state,input_observation=self.autoencoder(observation)
            curr_grad_step=0
            lane_angle=torch.Tensor([info]).to(device)
            state=torch.cat((state,lane_angle),dim=1).to(device)
            ep_reward=0
            while(1):
                #MSE loss step for autoencoder
                loss_ae=self.criterion_ae(input_observation,decoded_observation)
                # inp_show=np.uint8(input_observation.clone().detach().cpu().squeeze(0).permute(1,2,0)*255)
                # out_show=np.uint8(decoded_observation.clone().detach().cpu().squeeze(0).permute(1,2,0)*255)
                # filename=f"Image/{curr_grad_step}_in.jpg"
                # pyplot.imsave(filename, inp_show)
                # filename=f"Image/{curr_grad_step}_out.jpg"
                # pyplot.imsave(filename, out_show)
                # print(loss_ae)
                # assert 1==0
                mean_ae_loss+=loss_ae
                ae_steps+=1
                self.optimizer_autoencoder.zero_grad()
                loss_ae.backward()
                self.optimizer_autoencoder.step()

                # Calculate the mean and std dev of gaussian policy
                action_stats=self.policy_network(state)
                action_means=torch.tanh(action_stats[0])
                action_stddevs=torch.sigmoid(action_stats[1])#sigmoid so our std dev is between 0 and 1 and not negative or too large
                # Create a distribution from that mean and passed std_dev
                action_dist=Normal(loc=action_means,scale=action_stddevs)
                # Sample an action from the gaussian policy
                action=action_dist.sample()
                curr_grad_step+=1
                # Calculate log probability of that action
                action_logprob=action_dist.log_prob(action)
                action=action.detach().cpu().numpy()
                # action=[0.0,0.0]
                print(f"Info:{info*180}")
                next_observation, reward, terminated, info = self.env.step(action)
                # filename=f"Image/{curr_grad_step}_{action}.jpg"
                print(f"Gradstep, actopm, reward/{curr_grad_step}_{action},{reward}.jpg")
                print(f"Mean:{action_means}")
                # pyplot.imsave(filename, img_observation)
                img_observation=next_observation
                next_observation=torch.Tensor(next_observation).to(device)
                next_decoded_observation,next_state,next_input_observation=self.autoencoder(next_observation)
                lane_angle=torch.Tensor([info]).to(device)
                next_state=torch.cat((next_state,lane_angle),dim=1).to(device)
                cum_reward_stepsize+=reward
                ep_reward+=reward
                # The target for the critic, reward+V(next_state), and if our state is terminal, then it's only the reward
                critic_target=(reward+(1-terminated)*gamma*self.value_network(next_state)).detach()
                # The advantage function= critic_target-V(state)
                estimate=self.value_network(state)
                advantage=critic_target-estimate

                # if TRAIN_A2C:
                    # Critic Step

                self.optimizer_critic.zero_grad()
                critic_loss=torch.pow(advantage,2) #MSE for critic_loss
                mean_critic_loss+=critic_loss
                critic_steps+=1
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value_network.parameters(),max_norm=20.0)
                self.optimizer_critic.step()


                #Actor step
                self.optimizer_actor.zero_grad()
                actor_loss=-action_logprob*(advantage.detach()) #Loss=-1*action_logprob*advantage
                mean_actor_loss+=actor_loss
                actor_steps+=1
                actor_loss.sum().backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(),max_norm=10.0)
                self.optimizer_actor.step()

                print(f"Advantage:{advantage},estimation:{self.value_network(state)},critic_loss:{critic_loss},actor_loss:{actor_loss.sum()}")
                print("")

                # if (episode+1)%5==0:
                #     self.optimizer_critic.step()
                #     self.optimizer_critic.zero_grad()
                #     self.optimizer_actor.step()
                #     self.optimizer_actor.zero_grad()
                state=next_state
                observation=next_observation
                decoded_observation=next_decoded_observation
                input_observation=next_input_observation
                if terminated:
                    torch.save(self.policy_network.state_dict(), "policy.pth")
                    break

            #True False Changing
            # if (episode+1)==200:
            #     scheduler.step()
            #     TRAIN_A2C=not TRAIN_A2C
            #     print(f"Switching, now, TRAIN_AE:{TRAIN_AE}, TRAIN_A2C:{TRAIN_A2C}")




            #saving
            if ep_reward>max_ep_reward:
                print("Saving")
                torch.save(self.policy_network.state_dict(), "policy_best.pth")
                max_ep_reward=ep_reward
            #Logging
            if (episode+1)%reward_log_stepsize==0:
                avg_rewards.append(cum_reward_stepsize/reward_log_stepsize)
                mean_ae_loss=mean_ae_loss/ae_steps
                mean_critic_loss=mean_critic_loss/critic_steps
                avg_critics.append(mean_critic_loss)
                mean_actor_loss=mean_actor_loss/actor_steps
                print(f"Episode:{episode+1}, Average reward of last {reward_log_stepsize} steps: {cum_reward_stepsize/reward_log_stepsize}, MSE:{mean_ae_loss}, Average critic loss:{mean_critic_loss},Average actor loss:{mean_actor_loss}")
                mean_critic_loss=0
                mean_actor_loss=0
                critic_steps=0
                actor_steps=0
                mean_ae_loss=0
                ae_steps=0
                cum_reward_stepsize=0
        return self.policy_network,avg_rewards,avg_critics


def main():
    env=CarEnv()
    im_shape=(3,env.im_height,env.im_width)
    action_space_size=2
    hidden_size=400
    a2c=A2C(env=env,hidden_size=hidden_size,action_space_size=action_space_size)
    policy_SA,reward_history_SA,critic_history_SA=a2c.run_episodes(num_episodes=300,lr_actor=1e-4,lr_critic=1e-3,lr_autoencoder=1e-7,reward_log_stepsize=1,gamma=0.99)
    torch.save(policy_SA.state_dict(), "policy_SA_curvy_2.pth")
    np.save('reward_history_SA_curvy_2.npy',reward_history_SA)
    # np.save('critic_history_SA.npy',critic_history_SA)
    # torch.save(a2c.autoencoder.state_dict(), "ae.pth")

if __name__=='__main__':
    main()
