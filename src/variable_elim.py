"""
@Author: Andrea Minichova (s1021688)

Class for the implementation of the variable elimination algorithm.
"""

from pprint import pprint


class VariableElimination():

    def __init__(self, network):
        """
        Initialize the variable elimination algorithm with the specified network.
        Add more initializations if necessary.
        """
        self.network = network # object of class BayesNet
        self.factors = self.network.probabilities # dictionary (keys are variables)

        # Testing things out
        alarm = self.network.probabilities['Alarm'] # indexing key 'Alarm'

        print('-----------------')
        print(alarm)
        alarm.drop(alarm.index[alarm['Burglary'] == 'True'], inplace = True) # removing all rows where B=T
        print('-----------------')
        print(alarm)
        print('----------------')

    # def init_hidden_vars(self, query, observed):
    #     """ 
    #     Initialize the hidden variables based on given query and observed variables
    #     """
    #     hidden_vars = dict(self.network.probabilities)
    #     for o in observed:
    #         if o in hidden_vars.keys():
    #             hidden_vars.pop(o)
    #     hidden_vars.pop(query)
    #     self.hidden_vars = hidden_vars

    #     print(f'The hidden variables are: ', end=' ')
    #     for k in hidden_vars.keys():
    #         print(k, end=' ')

    def update_factors(self, observed):
        """ 
        Initialize factors, based on the observation
        """

        # TODO: rename factors as in dictionary "probabilities". Let each key be a tuple
        # of factor index and list of variables within the factor.
        # Index is necessary, if we ever encounter more factors with the same variables inside

        factors_copy = dict.copy(self.factors)
        for o in observed:
            for f in factors_copy:
                print(type(f.keys()))
                # for k in list(f.keys()):
                #     if o == k:
                #         print('----------------------------')
                #         print(f'{o} is present in table {f}')

    def run(self, query, observed, elim_order):
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a list specifying the elimination ordering
                        or a function that will determine an elimination ordering
                        given the network during the run

        Output: A variable holding the probability distribution
                for the query variable
        """

        print(f'The query variable is: {query}')
        if observed: 
            print(f'The observed variables are: {observed}')
        
        self.update_factors(observed)

        # TODO: Print formulas
        # print(f'Product formula to compute the query: P({query} | {observed}) = \u03A3_A,B,C ')


