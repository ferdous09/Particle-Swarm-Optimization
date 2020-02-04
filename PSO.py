"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created on Sat Nov 30 23:49:43 2019
@author: mdferdouspervej

This Code does: 
    Minimize: f(x1,x2) = (x2-x1)**4 + 12*x1*x2 - x1 + x2 - 3 using - 
                (a) PSO algorithm
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()
plt.close('all')
title_font = {'fontname':'Times New Roman', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} 
axis_font = {'fontname':'Times New Roman', 'size':'14'}
plt.rcParams["font.family"] = "Times New Roman"

#========================================= Problem 2 #=========================================
def objfunc(x1,x2):
    f = (x2-x1)**4 + 12*x1*x2 - x1 + x2 - 3
    return f

def Evaluate_Gen(pop):
    fit = np.zeros((len(pop),1))
    for i in range(len(pop)):
        fit[i,0] = objfunc(pop[i][0,0], pop[i][0,1])
    gen_best = min(fit)
    gen_worst = max(fit)
    gen_avg = np.sum(fit) / len(pop)
    
    return gen_best, gen_avg, gen_worst
#================================== PSO ==================================
a0, b0 = -10, 10
total_no_of_particle = 60
max_iter = 100
generation_best = []
generation_worst = []
generation_avg = []
        # ===== initialization =====
w, w_min, psi_1, psi_2 = 0.9, 0.4, 2, 2
gbest = np.matrix([[1e10, 1e10]])
particle_position, pbest_all, velocity = [], [], []
for i in range(total_no_of_particle):
    particle_i_pos = np.random.uniform(a0, b0, size = (1, 2))           # x_i~U(b_low, b_up)
    particle_position.append(particle_i_pos)
    pbest = particle_i_pos                                              # personal best
    pbest_all.append(pbest)
    if objfunc(pbest[0,0], pbest[0,1]) < objfunc(gbest[0,0], gbest[0,1]):
        gbest = pbest
    velo_i = np.random.uniform(-abs(b0 - a0), abs(b0 - a0), size = (1, 2))
    velocity.append(velo_i)

Gen_Evaluation = Evaluate_Gen(particle_position)
generation_best.append(float(Gen_Evaluation[0]))
generation_avg.append(float(Gen_Evaluation[1]))
generation_worst.append(float(Gen_Evaluation[2]))

iter_no = 1
del_w = (w - 0.4)/max_iter
        # Main Loop: PSO
while iter_no <= max_iter-1:
    for j in range(total_no_of_particle):
        for d in range(2):
            velocity[j][0, d] = w*velocity[j][0, d] + ( psi_1*np.random.uniform(0,1)*(pbest_all[j][0,d] - particle_position[j][0, d]) 
                    + psi_2*np.random.uniform(0,1)*(gbest[0,d] - particle_position[j][0, d]))
        particle_position[j] = particle_position[j] + velocity[j]
        
        if objfunc(particle_position[j][0,0], particle_position[j][0,1]) < objfunc(pbest_all[j][0,0], pbest_all[j][0,1]):
            pbest_all[j] = particle_position[j]
            if objfunc(pbest_all[j][0,0], pbest_all[j][0,1]) < objfunc(gbest[0, 0], gbest[0, 1]):
                gbest = pbest_all[j]
    Gen_Evaluation = Evaluate_Gen(particle_position)
    generation_best.append(float(Gen_Evaluation[0]))
    generation_avg.append(float(Gen_Evaluation[1]))
    generation_worst.append(float(Gen_Evaluation[2]))
    print('Generation no: ', iter_no, 'Obj Val: ', objfunc(gbest[0, 0], gbest[0, 1]))
    iter_no += 1
    w -= del_w 


fig, axes = plt.subplots(2,2)
axes[0,0].plot(np.arange(0, len(generation_avg)), generation_best, 'k-', label = 'Gen. Best')
axes[0,1].plot(np.arange(0, len(generation_avg)), generation_avg, 'b--', label = 'Gen. Average')
axes[1,0].plot(np.arange(0, len(generation_avg)), generation_worst, 'r-.', label = 'Gen. Worst')
axes[1,1].plot(np.arange(0, len(generation_avg)), generation_best, 'k-', label = 'Gen. Best')
axes[1,1].plot(np.arange(0, len(generation_avg)), generation_avg, 'b--', label = 'Gen. Average')
axes[1,1].plot(np.arange(0, len(generation_avg)), generation_worst, 'r-.', label = 'Gen. Worst')
axes[0,0].set_ylim(-10, max(generation_best) + 5)
axes[0,0].legend(loc = 0, ncol = 1)
axes[0,1].legend(loc = 0, ncol = 1)
axes[1,0].legend(loc = 0, ncol = 1)
axes[1,1].legend(loc = 0, ncol = 1)
#plt.tight_layout()
## Make Common X and Y Label
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.title('Particle Swarm Optimization: $(x_2-x_1)^4 + 12x_1x_2 - x_1 + x_2 - 3$', pad = 15)
plt.xlabel('Generation Number')
plt.ylabel('Objective Value')
plt.show()
#plt.savefig("Results/PSO.eps", dpi=1200, format='eps', orientation='portrait')


stop = timeit.default_timer()
print('Time: ', stop - start) 
