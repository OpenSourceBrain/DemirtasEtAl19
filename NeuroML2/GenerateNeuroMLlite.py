from neuromllite.NetworkGenerator import check_to_generate_or_run
from neuromllite import Simulation

from neuromllite import Network, Population, Projection, Cell, Synapse, InputSource, Input
from neuromllite import RandomConnectivity,RectangularRegion, RelativeLayout

import sys
import numpy as np


# This function generates the overview of the network using neuromllite
def internal_connections(pops, W):
    for pre in pops:
        for post in pops:

            weight = W[pops.index(post)][pops.index(pre)]
            print('Connection %s -> %s weight: %s'%(pre.id,
            post.id, weight))
            if weight!=0:

                net.projections.append(Projection(id='proj_%s_%s'%(pre.id,post.id),
                                                    presynaptic=pre.id,
                                                    postsynaptic=post.id,
                                                    synapse=syns[pre.id],
                                                    type='continuousProjection',
                                                    delay=0,
                                                    weight=weight,
                                                    random_connectivity=RandomConnectivity(probability=1)))


# Build the network
net = Network(id='Demirtas_corticalArea')
net.notes = 'Rate model with E and I populations'

net.parameters = { 'wee':      0,
                   'wei':      0.1,
                   'wie':      0,
                   'wii':      0}  
                   

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


input_source = InputSource(id='pulseGenerator0', 
                           neuroml2_input='PulseGenerator', 
                           parameters={'amplitude':'0nA', 'delay':'100.0ms', 'duration':'800.0ms'})
net.input_sources.append(input_source)

net.inputs.append(Input(id='stim',
                        input_source=input_source.id,
                        population=exc_pop.id,
                        percentage=100))
                        
exc_syn = Synapse(id='rsExc', lems_source_file='Demirtas_Parameters.xml')
inh_syn = Synapse(id='rsInh', lems_source_file='Demirtas_Parameters.xml')
net.synapses.append(exc_syn)
net.synapses.append(inh_syn)

syns = {exc_pop.id:exc_syn.id, inh_pop.id:inh_syn.id}
W = [['wee', 'wie'],
     ['wei','wii']]

# Add internal connections
pops = [exc_pop, inh_pop]
internal_connections(pops, W)

# Save to JSON format
net.id = 'Demirtas_network'
new_file = net.to_json_file('Demirtas_network.json')

sim = Simulation(id='SimDemirtas_network',
                                    duration='1000',
                                    dt='0.1',
                                    network=new_file,
                                    recordVariables={'r':{'all':'*'},
                                                     'e':{'all':'*'},
                                                     'f':{'all':'*'},
                                                     'iTotal':{'all':'*'},
                                                     'nu':{'all':'*'},
                                                     'internalNoise':{'all':'*'},
                                                     'S':{'all':'*'}}
                                    )
                            
sim.to_json_file('SimDemirtas_network.nmllite.json')

check_to_generate_or_run(sys.argv,sim)


      