import re
import pickle
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt


def get_period():
    return 10 

def plot_system():

    with open('sim_out.pkl', "rb") as f:
        pure_sim_data = pickle.load(f) 

    with open('sim_sampled_out.pkl','rb') as f:
        samp_sim_data = pickle.load(f)

    # Pure Data
    time    = pure_sim_data[0] 
    time    = np.array(time)/get_period()
    theta   = pure_sim_data[1]
    d_theta = pure_sim_data[2]
    u       = pure_sim_data[3] 

    # Sampled Data
    time_s    = samp_sim_data[0] 
    theta_s   = samp_sim_data[1] 
    d_theta_s = samp_sim_data[2] 
    u_s       = samp_sim_data[3] 

    time_s = np.array(time_s)/get_period()
    time_s = time_s     

    # Get absolute value and normalize between [0,0.5]
    norm_u = np.array(u_s) 
    norm_u = np.absolute(norm_u)
    max_u  = max(norm_u) 
    min_u  = min(norm_u)
    norm_u = 3 * (norm_u - min_u)/(max_u - min_u)
    

    plt.figure(1) 
    plt.plot(time, theta, 'b', linewidth=3.0)
    plt.step(time_s, theta_s, 'r',linewidth=3.0)
    plt.step(time_s, norm_u, 'g',linewidth=3.0) 
    plt.legend([r'$\theta(t)$',r'$\theta_s(t)$','$u(t)$'])

    return time_s


def get_rho_data(): 

    height_phi = 10
    sample_time = get_period() 
    sim_time    = 10 # seconds
    sim_res     = 1  # ms

    rho_idx  = [2,4,6,11] 


    number_data   = sim_time * 1000
    number_samples = int(number_data / sample_time) 


    rho  = [] 
    for i in range(0,len(rho_idx)):
        rho.append([])


    regex_1 = "(?<=\[).+?(?=\])" # Get number between brackets
    regex_2 = "(?<=NotNan\().+?(?=\))" # Get float 
    with open('monitor_out.txt') as f:
        for line in f:


            if("Terminating." in line):
                break

            output_num = re.search(regex_1, line)
            output_num = output_num.group(0) 
            output_num = int(output_num)

            if output_num == 0:
                print(line)
            if output_num in rho_idx:
                if output_num == 11: 
                    print(line)


                rho_val    = re.search(regex_2, line)
                rho_val    = rho_val.group(0)
                rho_val    = float(rho_val)

                rho[rho_idx.index(output_num)].append(rho_val)

    return rho


def plot_performance(time_s, rho):
    rho_0 = rho[0]
    rho_1 = rho[1]
    rho_2 = rho[2]
    rho_3 = rho[3]


    plt.figure(2)
    plt.step(time_s, rho_0, label="rho_0", linewidth=3.0)
    plt.title(r'$\rho_0$')

    plt.figure(3)
    plt.step(time_s, rho_1, label="rho_1", linewidth=3.0)
    plt.title(r'$\rho_1$')

    plt.figure(4)
    plt.step(time_s, rho_2, label="rho_2", linewidth=3.0)
    plt.title(r'$\rho_2$')

    plt.figure(5)
    plt.step(time_s, rho_3, label="rho_3", linewidth=3.0)
    plt.title(r'$\rho_{RT}$')



if __name__ == "__main__":

    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('font',**{'size':25})
    rc('text', usetex=True)

    rho = get_rho_data() 
    time_s = plot_system()
    plot_performance(time_s, rho)


    plt.show()


            




