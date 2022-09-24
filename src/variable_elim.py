"""
@Author: Andrea Minichova

Class for the implementation of the variable elimination algorithm
"""

import pprint
import pandas as pd
import itertools


class VariableElimination():

    def __init__(self, network):
        """
        Initialize the variable elimination algorithm with the factors of 'network'
        and the evidence
        """
        self.factors = network.probabilities
        self.observed = {}
        self.max_size_factor = 0


    def update_max_size_factor(self, factor):
        """"
        Updates the maximum factor size if applicable
        """
        if len(list(factor.columns[:-1])) > self.max_size_factor:
            self.max_size_factor = len(list(factor.columns[:-1]))


    def init_factors(self, observed):
        """ 
        Initialize factors, based on the observation 'observed'
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
        for key in self.factors:
            for o in observed:
                if o in key[1]:
                    self.factors[key].drop(self.factors[key].index[self.factors[key][o] != observed[o]],
                     inplace = True)

        # Save the size of biggest factor
        for key in self.factors:
            self.update_max_size_factor(self.factors[key])



    def get_factors(self, var):
        """ 
        Return (keys of) factors which contain variable 'var'
        """
        factors = []
        for key in self.factors:
            if var in key[1]:
                factors.append(key)
        return factors


    def generate_factor(self, vars):
        """
        Generates a factor (truth table) from 'vars' and initially dummy probabilities
        """
        # Generate truth table
        table = list(map(list, list(itertools.product(['True', 'False'], repeat=len(vars)))))
        row = []
        data = []
        for r in table:
            row = r
            row.append(0)     # dummy probability
            data.append(row)
        factor = pd.DataFrame(data, columns = vars + ['prob'])

        # In case of evidence, remove rows which do not correspond to evidence
        for o in self.observed:
            if o in vars:
                factor.drop(factor.index[factor[o] != self.observed[o]],
                    inplace = True)
        return factor


    def is_in(self, factor1, factor2):
        """
        Returns true if all varaible-value pairs from factor1 are present in factor2
        """
        vars = list(factor1.columns[:-1])
        for var in vars:
            if factor1.iloc[0][var] != factor2.iloc[0][var]:
                return False
        return True


    def multiply(self, factors):
        """"
        Return a factor (and its variables) which is a product of 'factors'
        """
        # Generate a new factor which we will populate with multiplied probabilities
        vars = []
        for key in factors:
            vars.extend(list(self.factors[key].columns[:-1]))
        vars = list(set(vars))
        product = self.generate_factor(vars)
        self.update_max_size_factor(product)

        # Multiply probabilities and return the final product
        probabilities = []
        product_prob = current_prob = 1
        for i in range (0,product.shape[0]):
            product_row = product.iloc[[i]]
            for key in factors:
                for j in range(0, self.factors[key].shape[0]):
                    current_row = self.factors[key].iloc[[j]]
                    if self.is_in(current_row, product_row):
                        current_prob = float(current_row.iloc[0]['prob'])
                        product_prob = product_prob * current_prob
                        break
            probabilities.append(product_prob)
            product_prob = 1   
        product['prob'] = probabilities
        return vars, product


    def can_sum_out(self, factor, i, j, vars):
        """ 
        Check whether row 'i' and row 'j' of 'factor' can be summed out, e.i. whether
        all values of 'vars' are the same in both rows
        """
        for var in vars:
            if factor.iloc[i][var] != factor.iloc[j][var]:
                return False
        return True


    def sum_out(self, var, factor):
        """"
        Return a factor (and its variables) in which 'var' was summed out of 'factor'
        """
        vars = [x for x in list(factor.columns[:-1]) if x != var]
        data = []
        for i in range (0,factor.shape[0]):
            for j in range (i+1,factor.shape[0]):
                if self.can_sum_out(factor, i, j, vars):
                    sum_prob = factor.iloc[i]['prob'] + factor.iloc[j]['prob']
                    row = []
                    for v in vars:
                        row.append(factor.loc[factor.index[i], v])
                    row.append(str(sum_prob))
                    data.append(row)
        sum_f = pd.DataFrame(data, columns = vars + ['prob'])
        return vars, sum_f


    def run(self, query, observed, elim_order):
        """
        Returns the probability distribution of 'query' variable 
        given the 'observed' variables and following the elimination order 'elim_order'
        """

        print('\n------------------------------------')
        print('|  Variable Elimination Algorithm  |')
        print('------------------------------------')

        print(f'\nA) The query variable: {query}\n')
        if observed: 
            self.observed = observed
            print(f'B) The observed variables: {observed}\n')
        else:
            print(f'B) There are no observed variables\n')
        self.init_factors(observed)
        print('D) The factors:\n')
        pprint.pprint(self.factors)
        print(f'\nE) The elimination ordering: {elim_order}\n')

        print(f'-----------------------------')
        print(f'|  F) The elimination loop  |')
        print(f'-----------------------------')

        it = 1
        i = len(self.factors) # Currently the highest factor index
        for v in elim_order:
            if v != query and v not in observed.keys():
                print(f'\nThe variable to eliminate: {v}')
                factors_with_v = self.get_factors(v)
                print('\nFactors to multiply:')
                print(factors_with_v)

                vars, mult_factor = self.multiply(factors_with_v)
                print('\nFactor after multiplication:')
                print(mult_factor)

                vars, sum_factor = self.sum_out(v, mult_factor)
                self.factors[i, tuple(vars)] = sum_factor

                print(f'\nFactor after summing out {v}:')
                print(sum_factor)
                for key in factors_with_v:
                    self.factors.pop(key)
                i += 1
                print(f'\nNew factors:')
                pprint.pprint(self.factors)

                it += 1
                print('\n-------------------')
                print(f'|  Next nr {it}:  |')
                print('-------------------')

        print('\nFactors to multiply:')
        print(self.factors.keys())
        vars, final_prob = self.multiply(self.factors.keys())
        print('\nFactor product after the final multiplication:')
        print(final_prob)

        print(f'\n-----------------------------------------------')
        print(f'|  G) The resulting CPT after normalization:  |')
        print(f'-----------------------------------------------')

        final_prob['prob'] = pd.to_numeric(final_prob['prob'], downcast="float")
        total = final_prob['prob'].sum()
        final_prob['prob'] = final_prob['prob'] / total
        print(f'\n {final_prob}')
        print(f'\nBiggest factor had size: {self.max_size_factor}')

        print(f'\nDone!')



