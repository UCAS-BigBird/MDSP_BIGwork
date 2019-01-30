import math
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from numpy import dot
from scipy.linalg import inv

#Q是process_var 过程噪声

def compute_dog_data(z_var, process_var, count=1, dt=1.):
    #"returns track, measurements 1D ndarrays"
    x, vel = 0., 1.
    z_std = math.sqrt(z_var)
    p_std = math.sqrt(process_var)
    xs, zs = [], []
    for _ in range(count):
        v = vel + (randn() * p_std)
        #这里速度是隐藏状态，而最后测量的距离是显示状态
        x +=math.pow(1.01,_)*v*dt
        xs.append(x)
        zs.append(x + randn() * z_std)
    return np.array(xs), np.array(zs)
    #ZS是观测值

dt = 1.
R_var = 10
Q_var = 0.01
x = np.array([[10.0, 4.5]]).T
P = np.diag([550, 40])
F = np.array([[1, dt],
              [0,  1]])
H = 1
R = np.array([[R_var]])
Q = Q_discrete_white_noise(dim=1, dt=dt, var=Q_var)
count=200
#track, zs = compute_dog_data(R_var, Q_var, count)


xs, cov = [], []

for z in zs:
    x = dot(F, x)
    P = dot(F, P).dot(F.T) + Q
    # update
    S = dot(H, P).dot(H.T) + R
    K = dot(P, H.T).dot(inv(S))
    y = z - dot(H, x)
    x += dot(K, y)
    P = P - dot(K, H).dot(P)

    xs.append(x)
    cov.append(P)

xs, cov = np.array(xs), np.array(cov)
xsflatten=np.array(xs.ravel())
x_new=[]
for i in range(count*2):
    if i%2==0:
        x_new.append(xsflatten[i])
#print(x_new)
plt.plot(zs,'r-')
plt.plot(track,'b+')
plt.plot(x_new,'k')
plt.grid()
plt.show()