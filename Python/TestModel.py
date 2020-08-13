
from hbnm.model.dmf import Model

import numpy as np

sc = np.array([[0.1,1],[0.5,0.1]])

print('Structural connectivity:\n%s'%sc)

model = Model(sc, g=0, norm_sc=True, hmap = None,
                 wee=(0.15, 0.), wei=(0.15, 0.),
                 syn_params=None, bold_params='obata',
                 verbose=True)
            
print('Created model')
t = 2

model.set_jacobian(compute_fic=True)

model.integrate(t,
                  dt=1e-4, n_save=10, stimulation=0.0,
                  delays=False, distance=None, velocity=None,
                  include_BOLD=True, from_fixed=True,
                  sim_seed=None, save_mem=False)
                 
print('Integrated model')

import matplotlib.pyplot as plt
print(model.sim.r_E)

plt.plot(model.sim.r_E.T)
plt.plot(model.sim.r_I.T)

plt.show()

