"""
@Author: Andrea Minichova (s1021688)

Variable Elimination Alogorithm

Framework provded by Joris van Vugt, Moira Berens, Leonieke van den Bulk
"""

from time import sleep
from bayesnet import BayesNet
from variable_elim import VariableElimination
import pprint
import itertools

if __name__ == '__main__':
    net = BayesNet('earthquake.bif')

    print('-----------------------------')
    print('|  Ntework specifications:  |')
    print('-----------------------------')
    
    print("\nNodes:")
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
    query = 'MaryCalls'

    # The evidence (can also be empty when there is no evidence)
    # evidence = {'JohnCalls' : 'True'}
    evidence = {}

    # The elimination ordering
    elim_order = net.nodes
    

    # # Testing different orderings:
    # orderings = itertools.permutations(['Alarm', 'Burglary', 'Earthquake', 'Johncalls'], 4)

    # for o in orderings:
    #     net = BayesNet('earthquake.bif')
    #     ve = VariableElimination(net)
    #     ve.run(query, evidence, o)
    #     print(list(o))
    #     sleep(10)


    # Run of the variable elimination algorithm for the queried node 
    # given the evidence and the elimination ordering
    ve.run(query, evidence, elim_order)
