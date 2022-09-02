import statsmodels.api as sm
from DLcausality.functions.data_generation import *
from DLcausality.functions.data_preprocessing import *
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random


class TE:
    def __init__(self, lag=0, dataset=pd.DataFrame([]), TE_list_linear=[], TE_list_shuffle_linear=[],
                 TE_list_nonlinear=[], TE_list_linear_con = [], TE_list_nonlinear_con = [], TE_list_shuffle_nonlinear=[],TE_list_shuffle_linear_con=[], TE_list_shuffle_nonlinear_con=[], z_scores_linear=[], z_scores_nonlinear=[], TE_list_lin_rev=[], TE_list_shuffle_lin_rev=[],
                 TE_list_nonlin_rev=[], TE_list_lin_con_rev = [], TE_list_nonlin_con_rev = [], TE_list_shuffle_nonlin_rev=[],TE_list_shuffle_lin_con_rev=[], TE_list_shuffle_nonlin_con_rev=[], z_scores_lin_rev=[], z_scores_nonlin_rev=[], signal=True):
        self.lag = lag
        self.dataset = dataset
        self.TE_list_linear = TE_list_linear
        self.TE_list_shuffle_linear = TE_list_shuffle_linear
        self.TE_list_nonlinear = TE_list_nonlinear
        self.TE_list_shuffle_nonlinear = TE_list_shuffle_nonlinear
        self.z_scores_linear = z_scores_linear
        self.z_scores_nonlinear = z_scores_nonlinear
        self.signal = signal
        self.TE_list_linear_con = TE_list_linear_con
        self.TE_list_nonlinear_con = TE_list_nonlinear_con
        self.TE_list_shuffle_linear_con = TE_list_shuffle_linear_con
        self.TE_list_shuffle_nonlinear_con = TE_list_shuffle_nonlinear_con
        self.TE_list_lin_rev = TE_list_lin_rev
        self.TE_list_shuffle_lin_rev = TE_list_shuffle_lin_rev
        self.TE_list_nonlin_rev = TE_list_nonlin_rev
        self.TE_list_lin_con_rev = TE_list_lin_con_rev
        self.TE_list_nonlin_con_rev = TE_list_nonlin_con_rev
        self.TE_list_shuffle_nonlin_rev = TE_list_shuffle_nonlin_rev
        self.TE_list_shuffle_lin_con_rev = TE_list_shuffle_lin_con_rev
        self.TE_list_shuffle_nonlin_con_rev = TE_list_shuffle_nonlin_con_rev
        self.z_scores_lin_rev = z_scores_lin_rev
        self.z_scores_nonlin_rev = z_scores_nonlin_rev

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

    def validation_Xpred(self, Xpred_old, Xpred_new):

        if Xpred_old.shape[1] == Xpred_new.shape[1]:
            self.signal = False
        return self.signal

    def linear_TE_XY(self, dataset, dependent_var="Y", splitting_percentage=0.7):

        training_set, testing_set = training_testing_set_linear(dataset, splitting_percentage)

        ols_joint = sm.OLS(training_set[dependent_var], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
        Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])

        if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged"]], Xpred):
            ypred = ols_joint.predict(Xpred)
            joint_residuals = testing_set[dependent_var] - ypred

            lag = "_lagged"
            dependent_var_lag = dependent_var + lag

            ols_independent = sm.OLS(training_set[dependent_var], sm.add_constant(training_set[dependent_var_lag])).fit()
            Xpred = sm.add_constant(testing_set[dependent_var_lag])
            ypred = ols_independent.predict(Xpred)
            independent_residuals = testing_set[dependent_var] - ypred

            TE = self.TE_calculation(joint_residuals, independent_residuals)

        else:
            TE = None

        return TE

    # def linear_TE_XY(self, dataset, splitting_percentage=0.7):
    #
    #     training_set, testing_set = training_testing_set_linear(dataset, splitting_percentage)
    #
    #     ols_joint = sm.OLS(training_set["Y"], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
    #     Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])
    #
    #     if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged"]], Xpred):
    #         ypred = ols_joint.predict(Xpred)
    #         joint_residuals = testing_set["Y"] - ypred
    #
    #         ols_independent = sm.OLS(training_set["Y"], sm.add_constant(training_set["Y_lagged"])).fit()
    #         Xpred = sm.add_constant(testing_set["Y_lagged"])
    #         ypred = ols_independent.predict(Xpred)
    #         independent_residuals = testing_set["Y"] - ypred
    #
    #         TE = self.TE_calculation(joint_residuals, independent_residuals)
    #
    #     else:
    #         TE = None
    #
    #     return TE
    #
    # def linear_TE_XY_reverse(self, dataset, splitting_percentage=0.7):
    #
    #     training_set, testing_set = training_testing_set_linear(dataset, splitting_percentage)
    #
    #     ols_joint_reverse = sm.OLS(training_set["X"], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
    #     Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])
    #
    #     if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged"]], Xpred):
    #         ypred = ols_joint_reverse.predict(Xpred)
    #         joint_residuals = testing_set["X"] - ypred
    #
    #         ols_independent_reverse = sm.OLS(training_set["X"], sm.add_constant(training_set["X_lagged"])).fit()
    #         Xpred = sm.add_constant(testing_set["X_lagged"])
    #         ypred = ols_independent_reverse.predict(Xpred)
    #         independent_residuals = testing_set["X"] - ypred
    #
    #         TE_reverse = self.TE_calculation(joint_residuals, independent_residuals)
    #
    #     else:
    #         TE_reverse = None
    #
    #     return TE_reverse

    # def linear_TE_XYZ(self, dataset, dependent_var="X", splitting_percentage=0.7):
    #
    #     training_set, testing_set = training_testing_set_linear(dataset, splitting_percentage)
    #
    #     ols_joint = sm.OLS(training_set["Y"], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
    #     Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])
    #
    #     if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged"]], Xpred):
    #         ypred = ols_joint.predict(Xpred)
    #         residuals_XY = testing_set["Y"] - ypred
    #
    #         ols_independent = sm.OLS(training_set["Y"], sm.add_constant(training_set["Y_lagged"])).fit()
    #         Xpred = sm.add_constant(testing_set["Y_lagged"])
    #         ypred = ols_independent.predict(Xpred)
    #         residuals_Y = testing_set["Y"] - ypred
    #
    #         transfer_entropy = self.TE_calculation(residuals_XY, residuals_Y)
    #
    #         ols_joint = sm.OLS(training_set["Y"], sm.add_constant(training_set[["Y_lagged", "X_lagged", "Z_lagged"]])).fit()
    #         Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged", "Z_lagged"]])
    #
    #         if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged", "Z_lagged"]], Xpred):
    #             ypred = ols_joint.predict(Xpred)
    #             residuals_XYZ = testing_set["Y"] - ypred
    #
    #             # Calculate Linear Transfer Entropy from Granger Causality
    #             transfer_entropy_con = self.TE_calculation(residuals_XYZ, residuals_XY)
    #
    #         else:
    #             transfer_entropy_con = None
    #     else:
    #         transfer_entropy = None
    #         transfer_entropy_con = None
    #
    #     return transfer_entropy, transfer_entropy_con

    def linear_TE_XYZ(self, dataset, dependent_var="Y", splitting_percentage=0.7):

        training_set, testing_set = training_testing_set_linear(dataset, splitting_percentage)

        ols_joint = sm.OLS(training_set[dependent_var], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
        Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])

        if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged"]], Xpred):
            ypred = ols_joint.predict(Xpred)
            residuals_XY = testing_set[dependent_var] - ypred

            lag = "_lagged"
            dependent_var_lag = dependent_var + lag

            ols_independent = sm.OLS(training_set[dependent_var], sm.add_constant(training_set[dependent_var_lag])).fit()
            Xpred = sm.add_constant(testing_set[dependent_var_lag])
            ypred = ols_independent.predict(Xpred)
            residuals_Y = testing_set[dependent_var] - ypred

            TE = self.TE_calculation(residuals_XY, residuals_Y)

            ols_joint = sm.OLS(training_set[dependent_var], sm.add_constant(training_set[["Y_lagged", "X_lagged", "Z_lagged"]])).fit()
            Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged", "Z_lagged"]])

            if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged", "Z_lagged"]], Xpred):
                ypred = ols_joint.predict(Xpred)
                residuals_XYZ = testing_set[dependent_var] - ypred

                # Calculate Linear Transfer Entropy from Granger Causality
                TE_con = self.TE_calculation(residuals_XYZ, residuals_XY)

            else:
                TE_con = None
        else:
            TE = None
            TE_con = None

        return TE, TE_con

    # def linear_TE_XYZ_reverse(self, dataset, splitting_percentage=0.7):
    #
    #     training_set, testing_set = training_testing_set_linear(dataset, splitting_percentage)
    #
    #     ols_joint_reverse = sm.OLS(training_set["X"], sm.add_constant(training_set[["Y_lagged", "X_lagged"]])).fit()
    #     Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged"]])
    #
    #     if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged"]], Xpred):
    #         ypred = ols_joint_reverse.predict(Xpred)
    #         residuals_XY = testing_set["X"] - ypred
    #
    #         ols_independent_reverse = sm.OLS(training_set["X"], sm.add_constant(training_set["X_lagged"])).fit()
    #         Xpred = sm.add_constant(testing_set["X_lagged"])
    #         ypred = ols_independent_reverse.predict(Xpred)
    #         residuals_Y = testing_set["X"] - ypred
    #
    #         TE_reverse = self.TE_calculation(residuals_XY, residuals_Y)
    #
    #         ols_joint_reverse = sm.OLS(training_set["X"], sm.add_constant(training_set[["Y_lagged", "X_lagged", "Z_lagged"]])).fit()
    #         Xpred = sm.add_constant(testing_set[["Y_lagged", "X_lagged", "Z_lagged"]])
    #
    #         if self.validation_Xpred(testing_set[["Y_lagged", "X_lagged", "Z_lagged"]], Xpred):
    #             ypred = ols_joint_reverse.predict(Xpred)
    #             residuals_XYZ = testing_set["X"] - ypred
    #
    #             # Calculate Linear Transfer Entropy from Granger Causality
    #             TE_con_reverse = self.TE_calculation(residuals_XYZ, residuals_XY)
    #
    #         else:
    #             TE_con_reverse = None
    #     else:
    #         TE_reverse = None
    #         TE_con_reverse = None
    #
    #     return TE_reverse, TE_con_reverse

    def shuffle_series(self, DF, only=None):

        if only is not None:
            shuffled_DF = DF.copy()
            for col in only:
                series = DF.loc[:, col].to_frame()
                shuffled_DF[col] = series.apply(np.random.permutation)
        else:
            shuffled_DF = DF.apply(np.random.permutation)

        return shuffled_DF

    def compute_z_scores_c(self, reverse=False):
        np.seterr(all='ignore')

        if reverse == False:
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

        if reverse == True:
            mean = np.mean(self.TE_list_shuffle_lin_rev)
            std = np.std(self.TE_list_shuffle_lin_rev)

            for TE in self.TE_list_lin_rev:
                z_score = (TE - mean) / std
                self.z_scores_lin_rev.append(z_score)

            mean = np.mean(self.TE_list_shuffle_nonlin_rev)
            std = np.std(self.TE_list_shuffle_nonlin_rev)

            for TE in self.TE_list_nonlin_rev:
                z_score = (TE - mean) / std
                self.z_scores_nonlin_rev.append(z_score)

    def compute_z_scores_t(self, reverse=False):
        np.seterr(all='ignore')

        if reverse == False:
            mean = np.mean(self.TE_list_shuffle_linear_con)
            std = np.std(self.TE_list_shuffle_linear_con)

            for TE in self.TE_list_linear_con:
                z_score = (TE - mean) / std
                self.z_scores_linear.append(z_score)

            mean = np.mean(self.TE_list_shuffle_nonlinear_con)
            std = np.std(self.TE_list_shuffle_nonlinear_con)

            for TE in self.TE_list_nonlinear_con:
                z_score = (TE - mean) / std
                self.z_scores_nonlinear.append(z_score)

        if reverse == True:
            mean = np.mean(self.TE_list_shuffle_lin_con_rev)
            std = np.std(self.TE_list_shuffle_lin_con_rev)

            for TE in self.TE_list_lin_con_rev:
                z_score = (TE - mean) / std
                self.z_scores_lin_rev.append(z_score)

            mean = np.mean(self.TE_list_shuffle_nonlin_con_rev)
            std = np.std(self.TE_list_shuffle_nonlin_con_rev)

            for TE in self.TE_list_nonlin_con_rev:
                z_score = (TE - mean) / std
                self.z_scores_nonlin_rev.append(z_score)

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
        transfer_entropy = self.TE_calculation(joint_residuals, independent_residuals)

        return transfer_entropy


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
        self.dataset = dataset
        self.update_dataset(dataset)

    def experiment(self, reverse=False, splitting_percentage=0.7):
        if reverse==False:
            dependent_var = "Y"
        else:
            dependent_var = "X"

        TE_linear = self.linear_TE_XY(self.dataset, dependent_var, splitting_percentage)

        if self.signal:

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset, reverse)
            TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle_linear = self.linear_TE_XY(dataset_shuffle, dependent_var, splitting_percentage)

            if self.signal:

                X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle, reverse)
                TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)


                if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False:
                    if TE_linear != None and TE_shuffle_linear != None and TE_nonlinear != None and TE_shuffle_nonlinear != None:

                        if reverse == False:
                            self.TE_list_linear.append(TE_linear)
                            self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                            self.TE_list_nonlinear.append(TE_nonlinear)
                            self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)
                        else:
                            self.TE_list_lin_rev.append(TE_linear)
                            self.TE_list_shuffle_lin_rev.append(TE_shuffle_linear)
                            self.TE_list_nonlin_rev.append(TE_nonlinear)
                            self.TE_list_shuffle_nonlin_rev.append(TE_shuffle_nonlinear)

        self.signal = True


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
        self.dataset = dataset
        self.update_dataset(dataset)

    def mean_of_diff_TE(self):
        md_TE_linear = np.mean(np.array(self.TE_list_linear_con) - np.array(self.TE_list_linear))
        # md_TE_shuffle_linear = np.mean(np.array(self.TE_list_shuffle_linear_con)-np.array(self.TE_list_shuffle_linear))
        md_TE_nonlinear = np.mean(np.array(self.TE_list_nonlinear_con) - np.array(self.TE_list_nonlinear))
        # md_TE_shuffle_nonlinear = np.mean(np.array(self.TE_list_shuffle_nonlinear_con) - np.array(self.TE_list_shuffle_nonlinear))

        return md_TE_linear, md_TE_nonlinear

    def multiple_experiment(self, num_exp, splitting_percentage=0.7):
        for i in range(num_exp):

            if i > 0:
                self.data_generation()

            TE_linear, TE_linear_con = self.linear_TE_XYZ(self.dataset, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset)
            TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(self.dataset)
            TE_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle_linear, TE_shuffle_linear_con = self.linear_TE_XYZ(dataset_shuffle, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle)
            TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(dataset_shuffle)
            TE_shuffle_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False and TE_nonlinear_con != False and TE_linear_con != False:
                self.TE_list_linear.append(TE_linear)
                self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                self.TE_list_nonlinear.append(TE_nonlinear)
                self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)

                self.TE_list_linear_con.append(TE_linear_con)
                self.TE_list_shuffle_linear_con.append(TE_shuffle_linear_con)
                self.TE_list_nonlinear_con.append(TE_nonlinear_con)
                self.TE_list_shuffle_nonlinear_con.append(TE_shuffle_nonlinear_con)


class TE_clm(TE):

    def __init__(self, X, Y, T, N, alpha, epsilon, r=4):
        TE.__init__(self, lag=1)
        self.T = T
        self.N = N
        self.alpha = alpha
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.r = r
        self.dist = "clm"

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

    def multiple_experiment(self, num_exp, splitting_percentage=0.7):
        for i in range(num_exp):

            if i > 0:
                self.varying_XY()
                self.data_generation()

            TE_linear = self.linear_TE_XY(self.dataset, splitting_percentage)

            if self.signal:
                X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset)
                TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

                dataset_shuffle = self.shuffle_series(self.dataset)
                TE_shuffle_linear = self.linear_TE_XY(dataset_shuffle, splitting_percentage)

                X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle)
                TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

                if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False:
                    self.TE_list_linear.append(TE_linear)
                    self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                    self.TE_list_nonlinear.append(TE_nonlinear)
                    self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)

            self.signal = True


class TE_tlm(TE):

    def __init__(self, X, Y, T, N, alpha, epsilon, r=4):
        TE.__init__(self, lag=1)
        self.T = T
        self.N = N
        self.alpha = alpha
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.r = r
        self.dist = "tlm"

    def data_generation(self):
        dataset = ternary_logistic_map(self.X, self.Y, self.T, self.N, self.alpha, self.epsilon)
        dataset["Y_lagged"] = dataset["Y"].shift(periods=self.lag)
        dataset["X_lagged"] = dataset["X"].shift(periods=self.lag)
        dataset["Z_lagged"] = dataset["Z"].shift(periods=self.lag)
        dataset = dataset.dropna(axis=0, how='any')
        self.dataset = dataset
        self.update_dataset(dataset)

    def mean_of_diff_TE(self):
        md_TE_linear = np.mean(np.array(self.TE_list_linear_con) - np.array(self.TE_list_linear))
        md_TE_nonlinear = np.mean(np.array(self.TE_list_nonlinear_con) - np.array(self.TE_list_nonlinear))

        return md_TE_linear, md_TE_nonlinear

    # def varying_XY(self):
    #     self.X = random.random()
    #     self.Y = random.random()

    def multiple_experiment(self, num_exp, splitting_percentage=0.7):
        for i in range(num_exp):

            if i > 0:
                # self.varying_XY()
                self.data_generation()

            TE_linear, TE_linear_con = self.linear_TE_XYZ(self.dataset, splitting_percentage)

            if self.signal:
                X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset)
                TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

                X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(self.dataset)
                TE_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

                dataset_shuffle = self.shuffle_series(self.dataset)
                TE_shuffle_linear, TE_shuffle_linear_con = self.linear_TE_XYZ(dataset_shuffle, splitting_percentage)

                X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle)
                TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

                X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(dataset_shuffle)
                TE_shuffle_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

                if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False and TE_nonlinear_con != False and TE_linear_con != False:
                    self.TE_list_linear.append(TE_linear)
                    self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                    self.TE_list_nonlinear.append(TE_nonlinear)
                    self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)

                    self.TE_list_linear_con.append(TE_linear_con)
                    self.TE_list_shuffle_linear_con.append(TE_shuffle_linear_con)
                    self.TE_list_nonlinear_con.append(TE_nonlinear_con)
                    self.TE_list_shuffle_nonlinear_con.append(TE_shuffle_nonlinear_con)

            self.signal = True
            print("The " + str(i+1) + " experiment is finished.")

# a = TE_cwp(T = 1, N = 500, alpha = 0.5, lag = 5, seed1=None, seed2=None)
# # a = TE_twp(T=1, N=100, alpha=0.5, phi=0.5, beta=0.5, lag=5, seed1=None, seed2=None, seed3=None)
# # a = TE_clp(X = 0.5, Y = 0.4, T = 1, N = 100, alpha = 0.2, epsilon = 0.4, r=4)
# # #
# # a = TE_tlm(X = 0.5, Y = 0.4, T = 1, N = 100, alpha = 0.2, epsilon = 0.4, r=4)
# a.data_generation()
# # print(a.linear_TE_XYZ(a.dataset, 0.7))
# a.multiple_experiment(5)
# a.compute_z_scores(a.dist)
# # # # # print(a.TE_list_)
# print(a.TE_list_shuffle_linear)
# print(a.TE_list_shuffle_nonlinear)
# print(np.std(a.TE_list_shuffle_linear))
# print(np.std(a.TE_list_shuffle_nonlinear))
# print(a.TE_list_linear)
# print(a.TE_list_nonlinear)
# print(a.z_scores_linear)
# # # # print(np.mean(a.z_scores_linear))
# print(a.z_scores_nonlinear)
# # # # print(np.mean(a.z_scores_nonlinear))
