import numpy as np
from scipy.io import savemat
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from network.net import PRLinear, PRLinearPlus
from utils.metrics import dist_real, dist_comp

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

class AForward(nn.Module):
    def __init__(self, in_c=None, out_c = None):
        super(AForward,self).__init__()
        self.lin1 = nn.Linear(in_c, out_c, bias=False)
        self.lin2 = nn.Linear(in_c, out_c, bias=False)
    def forward(self, x):
        x = x.reshape((1,-1,2))
        out_real = self.lin1(x[:,:,0])-self.lin2(x[:,:,1])
        out_comp = self.lin1(x[:,:,1]) + self.lin2(x[:,:,0])
        return torch.stack([out_real, out_comp],-1)

# define the problem
n = 400
OS = np.mgrid[1:3:0.1]
Test_os = len(OS)
Test_trial = 30
Recov_loss = np.zeros((Test_trial, Test_os,2))
Recov_rel = np.zeros((Test_trial, Test_os,2))

for ios, os in enumerate(OS):
    m = int(os*n)
    for i_try in range(Test_trial):
        print('The test no {:d} of {:d} with os={:f}'.format(i_try,Test_trial,os))
        # generate the truth signal:f
        x0 = torch.rand((1,2*n)).detach().type(dtype)
        A = AForward(in_c=n,out_c=m)
        A = A.type(dtype)
        y = A(x0)
        int_y = torch.sum(torch.pow(y,2),-1)
        y_obs = int_y.detach().type(dtype)

        x00 = x0.reshape((1,-1,2))

        # define the net and optim
        net = PRLinear(num_output_channels=2*n)
        net = net.type(dtype)

        optimizer = torch.optim.Adam([{'params':net.parameters()}], lr=0.01)
        # optimizer = torch.optim.SGD([{'params':net.parameters()}], lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

        mse = torch.nn.MSELoss().type(dtype)

        num_iter = 20000

        net_input = torch.rand((1,2*n)).type(dtype)
        net_input_saved = net_input.detach().clone()


        for step in range(num_iter):

            # input regularization
            net_input = net_input_saved #+ reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()


            # change the learning rate
            scheduler.step(step)
            optimizer.zero_grad()

            # get the network output
            out_x_m = net(net_input)

            out_y = A(out_x_m)
            int_y = torch.sum(torch.pow(out_y,2),-1)

            total_loss = mse(int_y,y_obs)
            
            total_loss.backward()
            optimizer.step()
            out_x_m_3 = out_x_m.reshape((1,-1,2))
            mse_l = dist_comp(out_x_m_3,x00)
            # mse_l = dist_real(out_x_m,x0)
        #     mse_l_0 = min(mse(out_x_m,x0).data,mse(-out_x_m,x0).data)
            if step % 400 == 0:
                print('Iter: {:5d}---error: {:8f}----rel : {:8f}'.format(step,total_loss.item(), mse_l))
            if total_loss.item() < 1e-12:
                break
        
        Recov_loss[i_try,ios,0] = total_loss.item()
        Recov_rel[i_try,ios,0] = mse_l

        net = PRLinear(num_output_channels=2*n,skip=False)
        net = net.type(dtype)

        optimizer = torch.optim.Adam([{'params':net.parameters()}], lr=0.01)
        # optimizer = torch.optim.SGD([{'params':net.parameters()}], lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

        mse = torch.nn.MSELoss().type(dtype)

        num_iter = 20000

        net_input = torch.rand((1,2*n)).type(dtype)
        net_input_saved = net_input.detach().clone()


        for step in range(num_iter):

            # input regularization
            net_input = net_input_saved #+ reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()


            # change the learning rate
            scheduler.step(step)
            optimizer.zero_grad()

            # get the network output
            out_x_m = net(net_input)

            out_y = A(out_x_m)
            int_y = torch.sum(torch.pow(out_y,2),-1)

            total_loss = mse(int_y,y_obs)
            
            total_loss.backward()
            optimizer.step()
            out_x_m_3 = out_x_m.reshape((1,-1,2))
            mse_l = dist_comp(out_x_m_3,x00)
            # mse_l = dist_real(out_x_m,x0)
        #     mse_l_0 = min(mse(out_x_m,x0).data,mse(-out_x_m,x0).data)
            if step % 400 == 0:
                print('Iter: {:5d}---error: {:8f}----rel : {:8f}'.format(step,total_loss.item(), mse_l))
            if total_loss.item() < 1e-12:
                break
        
        Recov_loss[i_try,ios,1] = total_loss.item()
        Recov_rel[i_try,ios,1] = mse_l
        # savemat('real_pr.mat',{'Recov_rel': Recov_rel,'Recov_loss':Recov_loss})
    savemat('comp_pr_skip.mat',{'Recov_rel': Recov_rel,'Recov_loss':Recov_loss})

savemat('comp_pr_skip.mat',{'Recov_rel': Recov_rel,'Recov_loss':Recov_loss})