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


    def training_testing_set(self, dataset, percentage):
        split_pos = round(dataset.shape[0] * percentage)
        training_set = dataset.iloc[0:split_pos]
        testing_set = dataset.iloc[split_pos:]
        return training_set, testing_set


    def linear_TE(self, dataset, dist, splitting_percentage = 0.7):

        training_set, testing_set = self.training_testing_set(dataset, splitting_percentage)

        if dist == "cwp" or "clp":
            ols_joint = sm.OLS(training_set["Y"], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
            Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])
            ypred = ols_joint.predict(Xpred)
            joint_residuals = testing_set["Y"] - ypred

            ols_independent = sm.OLS(training_set["Y"], sm.add_constant(training_set["Y_lagged"])).fit()
            Xpred = sm.add_constant(testing_set["Y_lagged"])
            ypred = ols_independent.predict(Xpred)
            independent_residuals = testing_set["Y"] - ypred

        if dist == "twp":

            ols_joint = sm.OLS(training_set["Y"], sm.add_constant(training_set[["Y_lagged", "X_lagged", "Z_lagged"]])).fit()
            Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged", "Z_lagged"]])
            ypred = ols_joint.predict(Xpred)
            joint_residuals = testing_set["Y"] - ypred

            ols_independent = sm.OLS(training_set["Y"], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
            Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])
            ypred = ols_independent.predict(Xpred)
            independent_residuals = testing_set["Y"] - ypred

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

    # def nonlinear_TE(self, X_ir, Y_ir, X_jr, Y_jr):
    #     independent_residuals = MLP(X_ir,Y_ir)
    #     joint_residuals =


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
        self.dist = "cwp"

    def data_generation(self):
        dataset = coupled_wiener_process(self.T, self.N, self.alpha, self.lag, self.seed1, self.seed2)
        dataset["Y_lagged"] = dataset["Y"].shift(periods=self.lag)
        dataset = dataset.dropna(axis=0, how='any')
        self.dataset = dataset
        self.update_dataset(dataset)

    def multiple_experiment_linearTE(self, num_exp, splitting_percentage = 0.7):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE = self.linear_TE(self.dataset, self.dist, splitting_percentage)
            self.TE_list.append(TE)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle = self.linear_TE(dataset_shuffle, self.dist, splitting_percentage)
            self.TE_list_shuffle.append(TE_shuffle)

    # def XY_MLP(self):


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
        self.dist = "twp"

    def data_generation(self):
        dataset = ternary_wiener_process(self.T, self.N, self.alpha, self.phi, self.beta, self.lag, seed1=None, seed2=None, seed3=None)
        dataset["Y_lagged"] = dataset["Y"].shift(periods=self.lag)
        dataset = dataset.dropna(axis=0, how='any')
        self.dataset = dataset
        self.update_dataset(dataset)

    def multiple_experiment_linearTE(self, num_exp, splitting_percentage = 0.7):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE = self.linear_TE(self.dataset, self.dist, splitting_percentage)
            self.TE_list.append(TE)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle = self.linear_TE(dataset_shuffle, self.dist, splitting_percentage)
            self.TE_list_shuffle.append(TE_shuffle)


class TE_clp(TE):

    def __init__(self, X, Y, T, N, alpha, epsilon, lag, r=4):
        TE.__init__(self, lag)
        self.T = T
        self.N = N
        self.alpha = alpha
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.r = r
        self.dist = "clp"

    def data_generation(self):
        dataset = coupled_logistic_map(self.X, self.Y, self.T, self.N, self.alpha, self.epsilon, self.r)
        dataset["Y_lagged"] = dataset["Y"].shift(periods=self.lag)
        dataset["X_lagged"] = dataset["X"].shift(periods=self.lag)
        dataset = dataset.dropna(axis=0, how='any')
        self.dataset = dataset
        self.update_dataset(dataset)

    def multiple_experiment_linearTE(self, num_exp, splitting_percentage = 0.7):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE = self.linear_TE(self.dataset, self.dist, splitting_percentage)
            self.TE_list.append(TE)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle = self.linear_TE(dataset_shuffle, self.dist, splitting_percentage)
            self.TE_list_shuffle.append(TE_shuffle)



a = TE_cwp(T = 1, N = 100, alpha = 0.5, lag = 5, seed1=None, seed2=None)
a.data_generation()
a.multiple_experiment_linearTE(10)
a.compute_z_scores()
# print(a.TE_list)
# print(a.TE_list_shuffle)
# print(a.z_scores)
print(np.mean(a.z_scores))