import sys
import time
import numpy as np
import pickle
from scipy.integrate import odeint 
import matplotlib.pyplot as plt

PERIOD = 250 # ms


# Globals
ms     = 0
SP     = 10
x      = 0
n      = 2 # Dimensions of \R^n
i = 0

u_list = []

def compute_control(x_vec):

    K   = np.array(np.mat('0.001,0.5'))  
    u = np.dot(-K,x_vec)

    return u


# Satellite as double integrator
def satellite_1(x,t):
    '''
    x: State vector as 1xn
    t: time
    '''
#    x_vec = x.transpose() # Make nx1 
    x_vec = x.reshape(n,1) # Make nx1 


    # Dynamics
    m = 100 # Mass kg
    l = 5   # length m
    d = 1
    I = (m*l**2)/12 

    A = np.array(np.mat('0, 1; 0, 0'))
    B = np.array([[0],[d/I]])


    # Controller
    ref = 0 # Theta SP
    u = compute_control(x_vec)
    
    x_dot = np.dot(A, x_vec) + np.dot(B, u) 
    x_dot = x_dot.flatten()
    x_dot = x_dot.tolist() 

    return x_dot 

# Single dimensional example
def f(x, t):
    a = 10
    b = 1

    ref = SP 
    K = 15 
    Nu = -a/b
    Nx = 1

    u = -K*x + (Nu + K*Nx)*ref 
    return a*x + b*u 



def get_period():
    """
    Returns the period in ms
    """
    return 250

def do_sampling(tspan, result):

    sim_out = open("sim_out.pkl", "wb")
    sim_sampled_out = open("sim_sampled_out.pkl", "wb")


    theta   = np.round(result[:,0],3)
    # Hack to get more like zero to 1 step response
    theta = theta + 10
    d_theta = np.round(result[:,1],3) #TODO: d_theta is wrong
    u       = theta.copy()
    for i in range(0,len(theta)):
        x_vec = np.array([[theta[i]],[d_theta[i]]]) 
        u[i] = compute_control(x_vec)
    

    # Sampled Data
    time_s    = []
    theta_s   = []
    d_theta_s = []
    u_s       = []


    # Output pure data for later reuse
    data_obj = (tspan, theta, d_theta, u)
    pickle.dump(data_obj, sim_out)

    # Output the simulation as sampled 
    print("{},{},{},{}".format("time","u","theta","d_theta"))

    i = 0
    for i in range(0,len(tspan)):
        T_s = get_period()


        if(i % T_s == 0):

            theta_i   = theta[i]
            d_theta_i = d_theta[i] 
            u_i       = u[i]
            time_i    = tspan[i]

            # Collect samples
            time_s.append(time_i)
            theta_s.append(theta_i)
            d_theta_s.append(d_theta_i)
            u_s.append(u_i)

            # Print 
            string = "{},{},{},{}".format(time_i, u_i, theta_i, d_theta_i)        
            print(string)
           

            time.sleep(T_s/1000)

    # Dump sampled data
    data_obj = (time_s,theta_s,d_theta_s,u_s)
    pickle.dump(data_obj, sim_sampled_out)

    sim_out.close()
    sim_sampled_out.close()
 

def do_plot(result):
    x    = result[:,0]

    plt.figure(1)
    plt.plot(tspan, x, 'b', linewidth=2, label='Controller') 
    plt.show()


if __name__ == "__main__":
 
    # Generate Dynamics
    sim_duration = 10  # seconds
    sim_resolution = 1 # ms
    num_steps = sim_duration * 1000

    x0 = [-10, 0] 
    tspan = np.linspace(0,num_steps,num_steps) # 1ms simulation 
    result = odeint(satellite_1, x0, tspan) 


    do_sampling(tspan,result)
    #do_plot(result) 




#
#    while(True):

#

#        #sys.stdout.write("{},{}".format(ms,x)) 
#        sys.stdout.flush()
#
#        x += 1
#        ms += PERIOD
#
#



        
