import numpy as np 
import matplotlib.pyplot as plt
import sympy as sp
import math

Δ = 0.01

x_g=0.8
y_g=0.6

x_k=0
y_k=0


x_error=x_k-x_g
y_error=y_k-y_g

tf=20
NN=tf / Δ
α = 0.2
γ = 0.9
R = [[0.05, 0], [0, 0.05]]
q = [[1, 0], [0, 1]]



def phi_basis(X1, X2):
    result = np.array([[X1, X2, X1*X1, X1*X2, X2*X2]]).T
    
    return result

def D_phi_basis(X1, X2):
    result = np.array([[1, 0, 2*X1, X2, 0],
                       [0, 1, 0, X1, 2*X2]]).T
    return (result)
    
NL = 5

W0 = np.array([[np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1)]]).T

U0 = np.array([[np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1)]]).T

def f(x_error, y_error): 
    result = np.array([[x_error],[y_error]])
    
    return result

def g(u):
    result = np.array([Δ,Δ])
    return result

def r(X1, X2, u):
    X = np.array([[X1],
                  [X2]])
    result = X.T @ q @ X + u.T @ R @ u
    return (result)


X_list = []
U_list = []
W_list = [W0]
u_list = []
u_k=1
W_k = W0
U_k = U0
kt = 0
uj_x = (-γ / 2)* np.linalg.inv(R) @ g(u_k) @ D_phi_basis(x_error, y_error).T @ W0 
uj_y=uj_x
u_j=np.array([uj_x, uj_y])

Wji = W0

tc = []
x_list, y_list, thetades_list = [],[],[]

w1_list, w2_list, w3_list, w4_list, w5_list = [],[],[],[],[]

rlist=[]

for k in range(int(NN)):
    
    u_list.append(u_j)
    x_list.append(x_error)
    y_list.append(y_error)
    X_k = f(x_error, y_error) + g(u_k)*u_j      
    X_list.append(X_k)
    x_error, y_error  = X_k[0][0], X_k[1][0]
    
    phi = phi_basis(x_error, y_error)
    
    r_k = r(x_error, y_error, u_j) + γ*float(phi_basis(x_error, y_error).T @ Wji)
   
    rlist.append(r_k)
    
    ecount = 1
    Wjim = W_list[-1]
    Wji = Wji - α*phi_basis(x_error, y_error)*(float(phi_basis(x_error, y_error).T @ Wji) - r_k)
    
    while ecount < 100 and float((Wji - Wjim).T @ (Wji - Wjim)) > 10**-5:
        Wjim = W_list[-1]
        #gradient descent
        Wji = Wji - α*phi_basis(x_error, y_error)*(float(phi_basis(x_error, y_error).T @ Wji) - r_k)
        ecount+=1 
        
        
    W_list.append(Wji)
    
    w1_list.append(Wji[0][0])
    w2_list.append(Wji[1][0])
    w3_list.append(Wji[2][0])
    w4_list.append(Wji[3][0])
    w5_list.append(Wji[4][0])
    
    
    #need to update u with the weights from W
    #policy improvement - update u_j
    u_j = -(γ/2)*np.linalg.inv(R) * g(u_k) @ D_phi_basis(x_error, y_error).T @ W_list[-1]
    u_list.append(u_j)
    tc.append(k)

# print(thetades_list)
# print(u_list)

# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], x_list[start_idx:end_idx], label = "x_error")
# plt.plot(tc[start_idx:end_idx], y_list[start_idx:end_idx], label = "y_error")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()

# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], thetades_list[start_idx:end_idx], label = "theta_des")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()

# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], w1_list[start_idx:end_idx], label = "w1")
# plt.plot(tc[start_idx:end_idx], w2_list[start_idx:end_idx], label = "w2")
# plt.plot(tc[start_idx:end_idx], w3_list[start_idx:end_idx], label = "w3")
# plt.plot(tc[start_idx:end_idx], w4_list[start_idx:end_idx], label = "w4")
# plt.plot(tc[start_idx:end_idx], w5_list[start_idx:end_idx], label = "w5")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()


# def error_to_trajectory(error_list):
#     result = []
#     for er in error_list:
#         result.append(er+x_g)
#     return result
# x_list = error_to_trajectory(x_list)
# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], x_list[start_idx:end_idx], label = "x")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()


# def error_to_trajectory(error_list):
#    result = []
#    for er in error_list:
#        result.append(er+y_g)
#    return result
# y_list = error_to_trajectory(y_list)
# start_idx = 0   
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], y_list[start_idx:end_idx], label = "y")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()


