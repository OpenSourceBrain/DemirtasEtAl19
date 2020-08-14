
import numpy as np
from hbnm.model.dmf import Model
from hbnm.io import Data

from hbnm.model.utils import linearize_map, normalize_sc

# helper method from https://github.com/murraylab/hbnm/blob/master/scripts/optimization.py#L54 
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


################################################################################
###  Load or generate the Structural connectivity matrix

output_dir = '.'
input_dir = '../../../../../../git/hbnm/data/'
data = Data(input_dir, output_dir)     
sc, hmap, fc_obj = load_data(data)         # Full scale network
sc = np.array([[0.001,1],[0.001,0.001]])   # 2 cortical areas
sc = np.array([[1e-12]])                   # 1 cortical area
    
num_areas = sc.shape[0]
print('Structural connectivity for %i areas (%s):\n%s'%(num_areas, sc.shape, sc))



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

model._sigma = 0  # Turn off noise!
'''
model._sigma = 0'''

print('External currents: %s'%model._I_ext)
#model._I0_E = np.array([0.3]*num_areas)
#model._I0_I = np.array([0.3]*num_areas)
print('Baseline currents E: %s'%model._I0_E)
print('Baseline currents I: %s'%model._I0_I)


################################################################################
### Check stability
''''''
model._w_EE = np.array([0]*num_areas)
model._w_EI = np.array([.1]*num_areas)
model._w_IE = np.array([0]*num_areas)
model._w_II = np.array([0]*num_areas)

print('Weight E->E: %s'%model._w_EE)
print('Weight E->I: %s'%model._w_EI)
print('Weight I->E: %s'%model._w_IE)
print('Weight I->I: %s'%model._w_II)

# if True, local feedback inhibition parameters (w^{IE}) are adjusted to set the firing rates of
# excitatory populations to ~3Hz
compute_fic = False
            
            
j = model.set_jacobian(compute_fic=compute_fic)
print('System stable: %s'%j)

print('Calc weight K E->E: %s'%model._K_EE)
print('Calc weight K E->I: %s'%model._K_EI)
print('Calc weight K I->E: %s'%model._K_IE)
print('Calc weight K I->I: %s'%model._K_II)


################################################################################
### Set initial values

#  If True, the simulation will begin using steady state values of the parameters
from_fixed = False
#from_fixed = True
           
if not from_fixed:
    
    import hbnm.model.params.synaptic as params
    model._r_E = np.array([ 3.5268949118680477 ] * num_areas) 
    model._r_I = np.array([ 6.339408113275469 ] * num_areas)
    model._S_E = np.array([ 0.18438852019969373 ] * num_areas)
    model._S_I = np.array([ 0.06339408113275538 ] * num_areas)
    model._I_E = np.array([ 0.382 ] * num_areas)
    model._I_I = np.array([ 0.26739999999999997 ] * num_areas)
    
    model._S_E = np.array([0.15] * num_areas)
    model._S_I = np.array([0.15] * num_areas)


################################################################################
### Integrate model

duration = 1  # sec
dt=1e-4       # sec

n_save=10
n_save = 1 # Save every point

model.integrate(duration,
                  dt=dt, n_save=n_save, stimulation=0.0,
                  delays=False, distance=None, velocity=None,
                  include_BOLD=True, from_fixed=from_fixed,
                  sim_seed=1234, save_mem=False)
                 
print(model._r_E)
print('Integrated model')

################################################################################
### Plot results

import matplotlib.pyplot as plt
print(model.sim.r_E)

var_types = {'Rates':['r_E','r_I'],
        'S variables':['S_E','S_I'],
        'Currents':['I_E','I_I']}
times = [i*dt for i in range(int(duration/dt) +1)]

for var_type in var_types:
    fig = plt.figure()
    fig.canvas.set_window_title(var_type)
    plt.title(var_type)
    for var in var_types[var_type]:
        a = eval('model.sim.%s'%var)
        for i in range(num_areas):
            print('Plotting %i values for %s: %s in area %i (%s -> %s)'%(len(a[i]), var_type, var, i, a[i][0], a[i][-1]))
            plt.plot(times, a[i],label='%s[%i]'%(var,i))
            plt.xlabel('Time (s)')
            
        f = open('%s_%iareas.dat'%(var, num_areas),'w')
        for ti in range(len(times)):
            f.write('%s\t'%(times[ti], ))
            for i in range(num_areas):
                f.write('%s\t'%(a[i][ti], ))
            f.write('\n')
        f.close()
        
    plt.legend()
    
plt.show()

