"""
@Author: Andrea Minichova (s1021688)

Class for the implementation of the variable elimination algorithm
"""

import pprint
import pandas as pd
import itertools


class VariableElimination():

    def __init__(self, network):
        """
        Initialize the variable elimination algorithm with the specified network
        """
        self.factors = network.probabilities


    def update_factors(self, observed):
        """ 
        Initialize factors, based on the observation observed
        """
        # Represent factors with keys with an index and a tuple of involved variables
        # 1. Index is useful once there are factors that contain the same variables
        # 2. Involved variables are useful for eliminating variables in the VE algorithm
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
        Return (keys of) factors which contain variable var
        """
        factors = []
        for key in self.factors:
            if var in key[1]:
                factors.append(key)
        return factors


    def multiply(self, factors):
        """
        Return a factor which is a result of multiplying all factors
        Factors is a list of keys with which we can access individual factors
        """
        vars = []
        for key in factors:
            vars.extend(list(self.factors[key].columns[:-1]))
        vars = list(set(vars))  # Remove duplicate variables
        product = self.generate_factor(vars)
        print('--------------------- NEW FACTOR ------------------------')
        print(product)

        probabilities = []
        prob = 1
        current_prob = 1

        # Loop through rows of factor which is eventually the final product
        for i in range (0,product.shape[0]):
            product_row = product.loc[product.index[i]]
            # Loop through all factors which need to be multiplied
            for key in factors:
                # Loop through each row of a given factor to match the variable-value pairs
                for j in range(0, self.factors[key].shape[0]):
                    row = self.factors[key].loc[self.factors[key].index[j]]
                    vars = list(row.keys()[:-1])
                    match = False
                    for var in vars:
                        print('hi')
                    # if matches with variable-value pairs from product factor, get its probability current_prob
                    prob = prob * current_prob
                    break
            probabilities.append(prob)
            current_prob = 1
            prob = 1

        # add the whole probabilities column!
        product['prob'] = probabilities
        return product
        


    # def mult_fact(self, f1, f2):
    #     """
    #     Returns a factor which is a result of multiplying two factors f1 and f2
    #     """
    #     # Find what variables are shared
    #     # Find rows in both factors where shared variables have the same values
    #     # Multiply probabilities, add to the new factor
    #     # Columns in the new factor have all shared variables (make sure they fit tho)
    #     data = []
    #     factor = pd.DataFrame(data, columns = '')
    #     return factor


    def generate_factor(self, vars):
        table = list(map(list, list(itertools.product(['True', 'False'], repeat=len(vars)))))
        row = []
        data = []
        for r in table:
            row = r
            row.append('0')
            data.append(row)
        new_factor = pd.DataFrame(data, columns = vars + ['prob'])
        return new_factor


    def sum_out(self, var, key):
        """"
        Return a factor in which var was summed out of factor with key
        """
        vars = [x for x in key[1] if x != var]
        factor = self.factors[key]
        print('---------------')
        print('Original factor')
        print(factor)
        print('---------------')
        data = []
        for i in range (0,factor.shape[0]):
            for j in range (1,factor.shape[0]-1):
                if i != j and self.can_sum_out(factor, i, j, vars):
                    if factor.loc[factor.index[i], var] != factor.loc[factor.index[j], var]:
                        print('-------------------------')
                        print('Summing out followin rows')
                        print(factor.loc[factor.index[i]])
                        print(factor.loc[factor.index[j]])
                        print('-------------------------')
                        sum_prob = factor.loc[factor.index[i], 'prob'] + factor.loc[factor.index[j], 'prob']
                        row = []
                        for v in vars:
                            row.append(factor.loc[factor.index[i], v])
                        row.append(str(sum_prob))
                        print(row)
                        data.append(row)

        new_factor = pd.DataFrame(data, columns = vars + ['prob'])
        print('----------------')
        print('Final factor is:')
        print(new_factor)
        return new_factor


    
    def can_sum_out(self, factor, i, j, vars):
        """ 
        Check whether row i and row j of factor can be summed out, e.i. whether
        all values of vars are the same in both rows
        """
        for v in vars:
            if factor.loc[factor.index[i], v] != factor.loc[factor.index[j], v]:
                return False
        return True


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
            print(f'\nThe variable to be eliminated: {v}')
            if v != query:
                v = 'Earthquake'
                factors_with_v = self.get_factors(v)
                new_factor1 = self.multiply(factors_with_v)
                # reduced_factor = self.sum_out(v, new_factor)
                # reduced_factor = self.sum_out(v, factors_with_v[1])
                # remove factors_with_v from self.factors, add reduced factor to self.factors (with new index i)
                i += 1
            break

        # Normalize
        print(f'-------------------------------------')
        print(f'G) The final CPT after normalization:')
        print(f'-------------------------------------')

        print(f'\nDone!')



