from neuromllite.NetworkGenerator import check_to_generate_or_run
from neuromllite import Simulation

from neuromllite import Network, Population, Projection, Cell, Synapse, InputSource, Input
from neuromllite import RandomConnectivity,RectangularRegion, RelativeLayout

import sys
import numpy as np


# Build the network
net = Network(id='Demirtas_corticalArea')
net.notes = 'Rate model with E and I populations'


r1 = RectangularRegion(id='Demirtas', x=0,y=0,z=0,width=1000,height=100,depth=1000)
net.regions.append(r1)

exc_cell = Cell(id='Exc', lems_source_file='Demirtas_Parameters.xml')
inh_cell = Cell(id='Inh', lems_source_file='Demirtas_Parameters.xml')
net.cells.append(exc_cell)
net.cells.append(inh_cell)

exc_pop = Population(id='Excitatory', 
                     size=1, 
                     component=exc_cell.id, 
                     properties={'color': '0.8 0 0','radius':10},
                     relative_layout = RelativeLayout(region=r1.id,x=-20,y=0,z=0))

inh_pop = Population(id='Inhibitory', 
                     size=1, 
                     component=inh_cell.id, 
                     properties={'color': '0 0 0.8','radius':10},
                     relative_layout = RelativeLayout(region=r1.id,x=20,y=0,z=0))

net.populations.append(exc_pop)
net.populations.append(inh_pop)



# Save to JSON format
net.id = 'Demirtas_network'
new_file = net.to_json_file('Demirtas_network.json')

sim = Simulation(id='SimDemirtas_network',
                                    duration='600',
                                    dt='0.01',
                                    network=new_file,
                                    recordVariables={'S':{'all':'*'}}
                                    )
                            
sim.to_json_file('SimDemirtas_network.nmllite.json')

check_to_generate_or_run(sys.argv,sim)


      