import numpy as np
import statsmodels.api as sm
from data_generation import *


class TE:
    def __init__(self, lag = 0, dataset = pd.DataFrame([]), TE_list =[], TE_list_shuffle=[], z_scores = []):
        self.lag = lag
        self.dataset = dataset
        self.TE_list = TE_list
        self.TE_list_shuffle = TE_list_shuffle
        self.z_scores = z_scores

    def update_dataset(self, new_dataset):
        self.dataset = new_dataset

    def TE_calculation(self, jr, ir):
        var_jr = np.var(jr)
        var_ir = np.var(ir)
        # Use Geweke's formula for Granger Causality
        granger_causality = np.log(var_ir / var_jr)

        # Calculate Linear Transfer Entropy from Granger Causality
        return granger_causality / 2


    def linear_TE(self, dataset, shuffle = "no", ternary = False):
        # Initialise list to return TEs
        if shuffle == "no":

            dataset["Y_lagged"] = dataset["Y"].shift(periods=self.lag)
            dataset = dataset.dropna(axis=0, how='any')
            self.dataset = dataset


        if ternary == False:
            joint_residuals = (sm.OLS(dataset["Y"], sm.add_constant(dataset[["Y_lagged", "X_lagged"]])).fit().resid)
            independent_residuals = (sm.OLS(dataset["Y"], sm.add_constant(dataset["Y_lagged"])).fit().resid)

        if ternary == True:
            joint_residuals = (sm.OLS(dataset["Y"], sm.add_constant(dataset[["Y_lagged", "X_lagged", "Z_lagged"]])).fit().resid)
            independent_residuals = (sm.OLS(dataset["Y"], sm.add_constant(dataset[["Y_lagged","X_lagged"]])).fit().resid)

        # Calculate Linear Transfer Entropy from Granger Causality
        transfer_entropies = self.TE_calculation(joint_residuals, independent_residuals)

        return transfer_entropies

    def shuffle_series(self, DF, only=None):

        if only is not None:
            shuffled_DF = DF.copy()
            for col in only:
                series = DF.loc[:, col].to_frame()
                shuffled_DF[col] = series.apply(np.random.permutation)
        else:
            shuffled_DF = DF.apply(np.random.permutation)

        return shuffled_DF

    def compute_z_scores(self):
        mean = np.mean(self.TE_list_shuffle)
        std = np.std(self.TE_list_shuffle)

        for TE in self.TE_list:
            z_score = (TE - mean)/std
            self.z_scores.append(z_score)


class TE_cwp(TE):

    def __init__(self, T, N, alpha, lag, seed1 = None, seed2 = None):
        TE.__init__(self, lag)
        self.T = T
        self.N = N
        self.alpha = alpha
        self.seed1 = seed1
        self.seed2 = seed2

    def data_generation(self):
        dataset = coupled_wiener_process(self.T, self.N, self.alpha, self.lag, self.seed1, self.seed2)
        self.update_dataset(dataset)

    def multiple_experiment(self, num_exp):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE = self.linear_TE(self.dataset)
            self.TE_list.append(TE)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle = self.linear_TE(dataset_shuffle, shuffle="yes")
            self.TE_list_shuffle.append(TE_shuffle)



class TE_twp(TE):

    def __init__(self, T, N, alpha, phi, beta, lag, seed1=None, seed2=None, seed3=None):
        TE.__init__(self, lag)
        self.T = T
        self.N = N
        self.alpha = alpha
        self.phi = phi
        self.beta = beta
        self.seed1 = seed1
        self.seed2 = seed2
        self.seed3 = seed3

    def data_generation(self):
        dataset = ternary_wiener_process(self.T, self.N, self.alpha, self.phi, self.beta, self.lag, seed1=None, seed2=None, seed3=None)
        self.update_dataset(dataset)

    def multiple_experiment(self, num_exp):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE = self.linear_TE(self.dataset, "no", True)
            self.TE_list.append(TE)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle = self.linear_TE(dataset_shuffle, shuffle="yes", ternary= True)
            self.TE_list_shuffle.append(TE_shuffle)



a = TE_twp(T=1, N=100, alpha=0.2, phi=0.5, beta=0.5, lag=5)
a.data_generation()
a.multiple_experiment(100)
a.compute_z_scores()
# print(a.TE_list)
# print(a.TE_list_shuffle)
# print(a.z_scores)
print(np.mean(a.z_scores))