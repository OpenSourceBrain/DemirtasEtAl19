
from hbnm.model.dmf import Model
from hbnm.io import Data

from hbnm.model.utils import linearize_map, normalize_sc

# from https://github.com/murraylab/hbnm/blob/master/scripts/optimization.py#L54 
def load_data(data):
    fin = data.load('demirtas_neuron_2019.hdf5')
    sc = fin['sc'].value
    fc = fin['fc'].value
    t1t2 = fin['t1wt2w'].value
    fin.close()

    # For left hemisphere, use first 180 indices
    sc = normalize_sc(sc[:180,:180])
    fc_obj = fc[:180,:180]
    hmap = linearize_map(t1t2[:180])

    return sc, hmap, fc_obj

import numpy as np

output_dir = '.'
input_dir = '../../../../../../git/hbnm/data/'


################################################################################
###  Load or generate the Structural connectivity matrix

data = Data(input_dir, output_dir)     
sc, hmap, fc_obj = load_data(data)         # Full scale network
sc = np.array([[0.001,1],[0.001,0.001]])   # 2 cortical areas
sc = np.array([[1e-12]])                   # 1 cortical area
    
num_pops = sc.shape[0]
print('Structural connectivity for %i pops (%s):\n%s'%(num_pops, sc.shape, sc))



################################################################################
### Create the model to execute

wee=(0.15, 0.)
wei=(0.15, 0.)
hmap = None # np.array([1])

model = Model(sc, g=0, norm_sc=True, hmap = hmap,
                 wee=wee, wei=wei,
                 syn_params=None, bold_params='obata',
                 verbose=True)
            
print('Created model')
      
        
################################################################################
### Change some parameters

model._sigma = 0
model._sigma = 0
print('External currents: %s'%model._I_ext)
model._I0_E = np.array([0.3]*num_pops)
model._I0_I = np.array([0.3]*num_pops)
print('Baseline currents E: %s'%model._I0_E)
print('Baseline currents I: %s'%model._I0_I)


################################################################################
### Check stability

j = model.set_jacobian(compute_fic=True)
print('System stable: %s'%j)

model._w_EE = np.array([0]*num_pops)
model._w_EI = np.array([0]*num_pops)
model._w_IE = np.array([0]*num_pops)
model._w_II = np.array([0]*num_pops)

print('Weight E->E: %s'%model._w_EE)
print('Weight E->I: %s'%model._w_EI)
print('Weight I->E: %s'%model._w_IE)
print('Weight I->I: %s'%model._w_II)


################################################################################
### Set initial values

#  If True, the simulation will begin using steady state values of the parameters
from_fixed = False
#from_fixed = True
           
if not from_fixed:
    model._r_E = np.array([5] * num_pops)
    model._r_I = np.array([5] * num_pops)
    model._I_I = np.array([0] * num_pops)
    model._I_E = np.array([0] * num_pops)
    model._S_I = np.array([0.15] * num_pops)
    model._S_E = np.array([0.15] * num_pops)


################################################################################
### Integrate model

duration = 1
dt=1e-4 

model.integrate(duration,
                  dt=dt, n_save=10, stimulation=0.0,
                  delays=False, distance=None, velocity=None,
                  include_BOLD=True, from_fixed=from_fixed,
                  sim_seed=1234, save_mem=False)
                 
print(model._r_E)
print('Integrated model')

################################################################################
### Plot results

import matplotlib.pyplot as plt
print(model.sim.r_E)

vars = {'Rates':['r_E','r_I'],
        'S variables':['S_E','S_I'],
        'Currents':['I_E','I_I']}

for v in vars:
    fig = plt.figure()
    fig.canvas.set_window_title(v)
    plt.title(v)
    for t in vars[v]:
        a = eval('model.sim.%s'%t)
        print(a.shape)
        print(a[0].shape)
        for i in range(num_pops):
            plt.plot(a[i],label='%s[%i]'%(t,i))
        
    plt.legend()
    
plt.show()

