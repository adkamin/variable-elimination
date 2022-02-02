"""
@Author: Andrea Minichova (s1021688)

Class for the implementation of the variable elimination algorithm.
"""

import pprint


class VariableElimination():

    def __init__(self, network):
        """
        Initialize the variable elimination algorithm with the specified network.
        Add more initializations if necessary.
        """

        self.factors = network.probabilities # dictionary (keys are variables)


    def update_factors(self, observed):
        """ 
        Initialize factors, based on the observation
        """

        # Represent factors with keys with an index and a tuple of involved variables
        # Index is useful once there are factors that contain the same variables
        # Involved variables are useful for eliminating variables in the VE algorithm
        factors_copy = dict.copy(self.factors)
        new_factors = {}
        i = 0
        for key in factors_copy:
            new_factors[(i, tuple(factors_copy[key].columns[:-1]))] = self.factors.pop(key)
            i += 1
        # Reduce the factors given the observation
        self.factors = new_factors
        for o in observed:
            for key in self.factors:
                if o in key[1]:
                    self.factors[key].drop(self.factors[key].index[self.factors[key][o] != observed[o]],
                     inplace = True)


    def get_factors(self, var):
        """ 
        Return factors which contain variable var
        """
        
        factors = []
        for key in self.factors:
            if var in key[1]:
                factors.append(self.factors[key])
        return factors


    def multiply(self, factors):
        """
        Return a factor which is product of factors
        """



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

        print('-------------------------------------------------------')
        print('Variable Elimination Algorithm with logs of steps taken')
        print('-------------------------------------------------------')

        print(f'\nA) The query variable: {query}\n')
        if observed: 
            print(f'B) The observed variables: {observed}\n')
        else:
            print(f'B) There are no observed variables\n')
        print('C) The formula to be computed: ...\n')
        self.update_factors(observed)
        print('D) The factors:\n')
        pprint.pprint(self.factors)
        print(f'\nE) The elimination ordering: {elim_order}\n')

        print(f'-----------------------')
        print(f'F) The elimination part')
        print(f'-----------------------')

        i = len(self.factors) # Currently the highest factor index
        for v in elim_order:
            print(f'\nVariable to be eliminated: {v}')
            if v != query:
                factors_with_v = self.get_factors(v)
                # new_factor = self.multiply(factors_with_v)
                # reduced_factor = self.sum_out(v, factors_with_v)
                # remove factors_with_v from self.factors, add reduced factor to self.factors (with new index)
                i += 1



