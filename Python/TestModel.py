
from hbnm.model.dmf import Model
from hbnm.io import Data


from hbnm.model.utils import subdiag, linearize_map, normalize_sc, fisher_z

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

sc = np.array([[0.001,1],[0.001,0.001]])
#sc = np.array([[1e-12]])
output_dir = '.'
input_dir = '../../../../../../git/hbnm/data/'
data = Data(input_dir, output_dir)
sc, hmap, fc_obj = load_data(data)
    

num_pops = sc.shape[0]

print('Structural connectivity for %i pops (%s):\n%s'%(num_pops, sc.shape, sc))

model = Model(sc, g=0, norm_sc=True, hmap = None,
                 wee=(0.15, 0.), wei=(0.15, 0.),
                 syn_params=None, bold_params='obata',
                 verbose=True)
                 
#model._sigma = 0
            
print('Created model')
t = .2

model.set_jacobian(compute_fic=True)

model.integrate(t,
                  dt=1e-4, n_save=10, stimulation=0.0,
                  delays=False, distance=None, velocity=None,
                  include_BOLD=True, from_fixed=True,
                  sim_seed=None, save_mem=False)
                 
print('Integrated model')

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

