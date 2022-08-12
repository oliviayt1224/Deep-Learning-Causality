import statsmodels.api as sm
from data_generation import *
from data_preprocessing import *
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random


class TE:
    def __init__(self, lag=0, dataset=pd.DataFrame([]), TE_list_linear=[], TE_list_shuffle_linear=[],
                 TE_list_nonlinear=[], TE_list_shuffle_nonlinear=[], z_scores_linear=[], z_scores_nonlinear=[]):
        self.lag = lag
        self.dataset = dataset
        self.TE_list_linear = TE_list_linear
        self.TE_list_shuffle_linear = TE_list_shuffle_linear
        self.TE_list_nonlinear = TE_list_nonlinear
        self.TE_list_shuffle_nonlinear = TE_list_shuffle_nonlinear
        self.z_scores_linear = z_scores_linear
        self.z_scores_nonlinear = z_scores_nonlinear

    def update_dataset(self, new_dataset):
        self.dataset = new_dataset

    def TE_calculation(self, jr, ir):

        var_jr = np.var(jr)
        var_ir = np.var(ir)
        # Use Geweke's formula for Granger Causality

        if var_jr > 0:
            granger_causality = np.log(var_ir / var_jr)
            return granger_causality / 2
        # Calculate Linear Transfer Entropy from Granger Causality
        else:
            return False

    def linear_TE(self, dataset, dist, splitting_percentage=0.7):

        training_set, testing_set = training_testing_set_linear(dataset, splitting_percentage)

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

    def compute_z_scores(self):
        mean = np.mean(self.TE_list_shuffle_linear)
        std = np.std(self.TE_list_shuffle_linear)

        for TE in self.TE_list_linear:
            z_score = (TE - mean)/std
            self.z_scores_linear.append(z_score)

        mean = np.mean(self.TE_list_shuffle_nonlinear)
        std = np.std(self.TE_list_shuffle_nonlinear)

        for TE in self.TE_list_nonlinear:
            z_score = (TE - mean)/std
            self.z_scores_nonlinear.append(z_score)

    def MLP(self, X, Y, percentage):

        training_X, testing_X, training_Y, testing_Y = training_testing_set_nonlinear(X, Y, percentage)
        dim = training_X.shape[1]
        model = keras.Sequential()
        model.add(layers.Dense(100, activation='relu', input_dim=dim))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(training_X, training_Y, validation_split=0.33, epochs=15, batch_size=10)
        ypred = model.predict(testing_X)
        resi = testing_Y - ypred

        return resi

    def nonlinear_TE(self, X_jr, Y_jr, X_ir, Y_ir, percentage=0.7):

        joint_residuals = self.MLP(X_jr, Y_jr, percentage)
        independent_residuals = self.MLP(X_ir, Y_ir, percentage)
        transfer_entropies = self.TE_calculation(joint_residuals, independent_residuals)

        return transfer_entropies


class TE_cwp(TE):

    def __init__(self, T=1, N=200, alpha=0.5, lag=5, seed1=None, seed2=None):
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

    def multiple_experiment(self, num_exp, splitting_percentage=0.7):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE_linear = self.linear_TE(self.dataset, self.dist, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset)
            TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle_linear = self.linear_TE(dataset_shuffle, self.dist, splitting_percentage)
            if TE!=False:
                self.TE_list_shuffle_linear.append(TE_shuffle_linear)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle)
            TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            if TE_linear != False and TE_shuffle_linear != False:
                self.TE_list_linear.append(TE_linear)
                self.TE_list_shuffle_linear.append(TE_shuffle_linear)

            if TE_nonlinear != False and TE_shuffle_nonlinear != False:
                self.TE_list_nonlinear.append(TE_nonlinear)
                self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)


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

    def multiple_experiment(self, num_exp, splitting_percentage=0.7):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE_linear = self.linear_TE(self.dataset, self.dist, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(self.dataset)
            TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle_linear = self.linear_TE(dataset_shuffle, self.dist, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(dataset_shuffle)
            TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False:
                self.TE_list_linear.append(TE_linear)
                self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                self.TE_list_nonlinear.append(TE_nonlinear)
                self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)


class TE_clp(TE):

    def __init__(self, X, Y, T, N, alpha, epsilon, r=4):
        TE.__init__(self, lag=1)
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

    def varying_XY(self):
        self.X = random.random()
        self.Y = random.random()

    def varying_coeff(self):
        self.alpha = random.random()
        self.epsilon = random.random()

    def multiple_experiment(self, num_exp, splitting_percentage=0.7, varying_choice = "XY"):
        for i in range(num_exp):

            if i > 0:
                if varying_choice == "XY":
                    self.varying_XY()
                    self.data_generation()
                if varying_choice == "coeff":
                    self.varying_coeff()
                    self.data_generation()

            TE_linear = self.linear_TE(self.dataset, self.dist, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset)
            TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle_linear = self.linear_TE(dataset_shuffle, self.dist, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle)
            TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False:
                self.TE_list_linear.append(TE_linear)
                self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                self.TE_list_nonlinear.append(TE_nonlinear)
                self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)



# # a = TE_cwp(T = 1, N = 100, alpha = 0.5, lag = 5, seed1=None, seed2=None)
# # a = TE_twp(T=1, N=100, alpha=0.5, phi=0.5, beta=0.5, lag=5, seed1=None, seed2=None, seed3=None)
# a = TE_clp(X = 0.5, Y = 0.4, T = 1, N = 100, alpha = 0.2, epsilon = 0.4, r=4)

a = TE_clp(X = 0.5, Y = 0.4, T = 1, N = 300, alpha = 0.4, epsilon = 0.9, r=4)
a.data_generation()
a.multiple_experiment(100)
a.compute_z_scores()
# # print(a.TE_list_)
# # print(a.TE_list_shuffle)
# # print(a.z_scores_linear)
print(np.mean(a.z_scores_linear))
# # print(a.z_scores_nonlinear)
print(np.mean(a.z_scores_nonlinear))
