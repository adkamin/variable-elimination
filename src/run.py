"""
@Author: Andrea Minichova (s1021688)

Variable Elimination Alogorithm

Framework provded by Joris van Vugt, Moira Berens, Leonieke van den Bulk
"""

from bayesnet import BayesNet
from variable_elim import VariableElimination
import pprint

if __name__ == '__main__':
    net = BayesNet('earthquake.bif')
    
    print("Nodes:")
    print(f'{net.nodes}\n')
    print("Values:")
    print(f'{net.values}\n')
    print("Parents:")
    print(f'{net.parents}\n')
    print("Probabilities:")
    pprint.pprint(net.probabilities)

    # Instance of variable elimination algorithm
    ve = VariableElimination(net)

    # Node to be queried
    query = 'Alarm'

    # The evidence (can also be empty when there is no evidence)
    evidence = {'Burglary': 'True'}
    # evidence = {}

    # The elimination ordering
    elim_order = net.nodes

    # Run of the variable elimination algorithm for the queried node 
    # given the evidence and the elimination ordering
    ve.run(query, evidence, elim_order)
