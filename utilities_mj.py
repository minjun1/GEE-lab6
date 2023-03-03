import numpy as np
import math
from math import log10, sqrt 
#import cv2 
import torch
#from utils.inpainting_utils import *
import random

def RSNR(original, compressed): 
    mse = np.sum((original - compressed) ** 2) 

    mse2 = np.sum((original) ** 2) 

    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    rsnr = 10 * log10(mse2 / mse) 
    return rsnr 

def Mask(d, p, reg):
    # d = original data
    # p = percentage of missing traces 
    # option = regular / irregular missing 
    # reg : if reg=1, regular mask / if reg = 0, irregular mask 
    
    
    trace_num = len(d[0,0,:])
    sample_num = len(d[0,:,0])
    o = int(np.round(p/100*trace_num))
    
    if reg==1:

        if p>=50:
            mask=np.ones(shape=[1,sample_num,trace_num])
            remain_num = trace_num - o
            m = int(math.ceil(trace_num/remain_num))

            for i in range(remain_num):
                mask[0,:,i*m+1:i*m+m]=0


        if p<50:

            mask=np.zeros(shape=[1,sample_num,trace_num])
            remain_num = o
            m = int(math.ceil(trace_num/remain_num))

            for i in range(remain_num):
                mask[0,:,i*m+1:i*m+m]=1  
    if reg==0:
        
        ans=np.array(np.arange(trace_num))
        np.random.shuffle(ans)
        mask = np.ones(shape=[1,sample_num ,trace_num])
        
        
        mask[0,:,ans[:o]]=0

    mask[0,:,trace_num-1:]=1
    mask[0,:,0]=1
    
    return mask 

def threshold(img_fft, th, dim=2):

    p = (img_fft>th).float()
    m = (img_fft<-th).float()
    thred_comp = img_fft*p+img_fft*m
    img_thredi = torch.irfft(thred_comp, dim, onesided=False)
    
    return img_thredi


def pocs1(img_var, mask_var, th,alp = 0.2, dim=2):
    mask = torch_to_np(mask_var)
    img_fft = torch.rfft(img_var*mask_var, dim, onesided=False)
    
    test = threshold(img_fft,th,dim)
    pocs = torch_to_np(test)*(1-alp*mask)
    res = alp*torch_to_np(img_var)*mask+pocs

    return res

def pocs2(out, img_var, mask_var, th,alp = 0.2, dim=2):
    mask = torch_to_np(mask_var)
    img_fft = torch.rfft(out, dim, onesided=False)
    
    test = threshold(img_fft,th,dim)
    pocs = torch_to_np(test)*(1-alp*mask)
    res = alp*torch_to_np(img_var)*mask+pocs

    return res

def pocsn(data_n, th,alp = 0.2, dim=2):
    img_fft = torch.rfft(data_n, dim, onesided=False)
    test = threshold(img_fft,th,dim)
    pocs = torch_to_np(test)
    
    return pocs


def pocs_dip(out, img_var, mask_var, th,alp = 0.2, dim=2):
    mask = mask_var
    img_fft = torch.rfft(out, dim, onesided=False)
    test = threshold(img_fft,th,dim)
    pocs = test*(1-alp*mask)
    res = alp*img_var*mask+pocs

    return res

def tpow(inp,alp): 
    _, tt, _ = np.shape(inp)
    xx = np.zeros_like(inp)
    for i in range(tt):
        t = (i+1)*0.001
        xx[0,i,:]=t**alp*inp[0,i,:]
    return xx

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False
        
def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input
def get_noise3d(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1], spatial_size[2]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params
def get_params2(opt_over, net, net_input1,net_input2, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input1.requires_grad = True
            params += [net_input1]
            net_input2.requires_grad = True
            params += [net_input2]          
        else:
            assert False, 'what is it?'
            
            
def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    
        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)

        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False
        
def data_parallel(module, input, device_ids, output_device):
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)