"""
Functions for generating broadband and monochromatic planewaves
@author: Joseph Jennings
@version: 2020.04.04
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sciim

def line(nx,dx,nt,dt,v,t0,sign,ox=0.0,ot=0.0,kind='linear'):
  """
  Models a line given the velocity (slope) and a t0

  Parameters
    nx   - number of lateral samples
    dx   - the lateral spacing
    nt   - number of vertical samples
    dt   - the temporal sampling
    v    - velocity or slope of the line
    t0   - intercept of the line on the time axis
    sign - the direction of the slope of the line [1,-1]
    kind - the type of interpolation for applying the shift ['linear']

  Returns an array with a line
  """
  samp = int(t0/dt)
  if(samp >= nt):
    raise Exception("Please choose a t0 that is less than the total time")
  spk = np.zeros(nt)
  spk[samp] = 1
  rpt = np.tile(spk,(nx,1)).T
  x   = np.linspace(ox,dx*(nx-1),nx)
  tsq = sign * x/v
  shift  = tsq/dt
  ishift = shift.astype(int)

  if(kind == 'linear'):
    lmo = np.array([sciim.interpolation.shift(rpt[:,ix],shift[ix],order=1) for ix in range(nx)]).T
  else:
    lmo = np.array([np.roll(rpt[:,ix],ishift[ix]) for ix in range(nx)]).T

  return lmo

def ricker(nt,dt,f,amp,dly):
  """ 
  Given a time axis, dominant frequency and amplitude,
  returns a ricker wavelet

  Parameters
    nt  - length of output wavelet
    dt  - sampling rate of wavelet
    f   - dominant frequency of ricker wavelet
    amp - maximum amplitude of wavelet
    dly - time delay of wavelet

  Returns a 1D ricker wavelet
  """
  src = np.zeros(nt,dtype='float32')
  for it in range(nt):
    t = it*dt - dly
    pift = (np.pi*f*t)**2
    src[it] = amp*(1-2*pift)*np.exp(-pift)

  return src

def planewave(nt,dt,nx,dx,w,theta,ot=0.0,ox=0.0):
  """
  Generates a monochromatic plane wave with a specific
  angle (theta) and frequency w

  Parameters
    nt    - number of time samples
    dt    - sampling rate of the image
    nx    - number of lateral samples
    dx    - lateral sampling
    w     - frequency of plane wave
    theta - orientation of plane wave 
            [choose a range of angles between -0.1 and 0.1 for larger angles 
            the temporal frequency of the plane wave will increase]
    ot    - temporal origin [0.0]
    ox    - lateral origin [0.0]
  """
  u = w/np.sqrt(1 + np.tan(theta*np.pi/180.0)**2)
  v = np.tan(theta*np.pi/180.0)*u
  wave = np.zeros([nt,nx])
  for it in range(nt):
    t = ot + it*dt
    for ix in range(nx):
      x = ox + ix*dx
      wave[it,ix] = np.cos(2*np.pi*(u*t + v*x))

  return wave

def ellipsemask(n1,n2):
  """
  Creates an elliptical mask like what is shown in GIEE

  Parameters
    n1 - length of the slow axis
    n2 - length of the fast axis
  """
  maskout = np.ones([n1,n2])
  for i1 in range(n1):
    for i2 in range(n2):
      x = (i1-11.0)/n1 - 0.5
      y = (i2+9.0)/n2 - 0.3
      u = x+y
      v = (x-y)/2.0
      if(u**2 + v**2 < .15):
        maskout[i1,i2] = 0.0

  return maskout

def fft2d(img,d1,d2):
  """ Computes the fft of the image as well as the frequencies """
  n1 = img.shape[1]; dk1 = 1/(n1*d1); ok1 = -dk1*n1/2.0;
  n2 = img.shape[0]; dk2 = 1/(n2*d2); ok2 = -dk2*n2/2.0;
  k1 = np.linspace(ok1, ok1+(n1-1)*dk1, n1) 
  k2 = np.linspace(ok2, ok2+(n2-1)*dk2, n2) 
  imgfft = np.fft.fftshift(np.fft.fft2(img))
  
  return imgfft,k1,k2

