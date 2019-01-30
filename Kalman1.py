# from filterpy.kalman import MerweScaledSigmaPoints
# points=MerweScaledSigmaPoints(n=2,alpha=.1,beta=2.,kappa=1.)
# d=points.sigma_points(x=[0.,0],P=[[1.,.1],[.1,1]])
# print(points)
# print(d)
# # n=1
# # sigmas = np.zeros((2*n+1, n))
# # lambda_ = alpha**2 * (n + kappa) - n
# # Wc = np.full(2*n + 1,  1. / (2*(n + lambda_))
# # Wm = np.full(2*n + 1,  1. / (2*(n + lambda_))
# # Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
# # Wm[0] = lambda_ / (n + lambda_)
# #
# # U = scipy.linalg.cholesky((n+lambda_)*P) # sqrt
# # sigmas[0] = X
# # for k in range (n):
# #     sigmas[k+1]   = X + U[k]
# #     sigmas[n+k+1] = X - U[k]

import pandas as pd
import numpy
import matplotlib.pyplot as plt

c=pd.read_csv("C:/Users/UCAS_BigBird/Desktop/kalman_ou.csv")
z=c.values.ravel()
n_iter = c.values.shape[0]
sz = (n_iter,)  # size of array
Q = 2 # process variance
# 分配数组空间
xhat = numpy.zeros(sz)  # a posteri estimate of x 滤波估计值
P = numpy.zeros(sz)  # a posteri error estimate滤波估计协方差矩阵
xhatminus = numpy.zeros(sz)  # a priori estimate of x 估计值
Pminus = numpy.zeros(sz)  # a priori error estimate估计协方差矩阵
K = numpy.zeros(sz)  # gain or blending factor卡尔曼增益

R = 49 # estimate of measurement variance, change to see effect
# intial guesses
xhat[0] =0
P[0] = 0
K[0] =1

def rmse(predictions, targets):
    return numpy.sqrt(((predictions - targets) ** 2).mean())

for k in range(1, n_iter):
    # 预测
    xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
    Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
    # 更新
    K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P[k] = (1 - K[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

#xhatt=xhat.reshape(1,200)
print(xhat.shape)
print(z.shape)
dd=rmse(xhat,z)
print(dd)
plt.figure(figsize=(16,16),dpi=80)
#ax1 = plt.subplot(211)
plt.plot(xhat,'r-')
plt.plot(z,'ko')
plt.show()
#plt.plot(Pminus)
#plt.show()