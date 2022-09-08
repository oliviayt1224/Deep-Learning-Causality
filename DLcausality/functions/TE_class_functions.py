import statsmodels.api as sm
from DLcausality.functions.data_generation import *
from DLcausality.functions.data_preprocessing import *
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random


class TE:
    """Parent class TE"""

    def __init__(self, lag=0, dataset=pd.DataFrame([]), TE_list_linear=[], TE_list_shuffle_linear=[],
                 TE_list_nonlinear=[], TE_list_linear_con = [], TE_list_nonlinear_con = [],
                 TE_list_shuffle_nonlinear=[], TE_list_shuffle_linear_con=[], TE_list_shuffle_nonlinear_con=[],
                 z_scores_linear=[], z_scores_nonlinear=[], TE_list_lin_rev=[], TE_list_shuffle_lin_rev=[],
                 TE_list_nonlin_rev=[], TE_list_lin_con_rev = [], TE_list_nonlin_con_rev = [],
                 TE_list_shuffle_nonlin_rev=[], TE_list_shuffle_lin_con_rev=[], TE_list_shuffle_nonlin_con_rev=[],
                 z_scores_lin_rev=[], z_scores_nonlin_rev=[], signal=True):

        """ Initialize an instance for class TE.

        Parameters
        ----------
        lag : `int`, optional
            time lag.
        dataset : `pandas.DataFrame`, optional
            the dataset used for analysis.
        TE_list_linear : `list`, optional
            list of transfer entropy calculated from linear method.
        TE_list_shuffle_linear  : `list`, optional
            list of transfer entropy calculated from linear method using shuffled data.
        TE_list_nonlinear  : `list`, optional
            list of transfer entropy calculated from nonlinear method.
        TE_list_linear_con  : `list`, optional
            list of conditional transfer entropy calculated from linear method.
        TE_list_nonlinear_con  : `list`, optional
            list of conditional transfer entropy calculated from nonlinear method.
        TE_list_shuffle_nonlinear  : `list`, optional
            list of transfer entropy calculated from nonlinear method using shuffled data.
        TE_list_shuffle_linear_con  : `list`, optional
            list of conditional transfer entropy calculated from linear method using shuffled data.
        TE_list_shuffle_nonlinear_con  : `list`, optional
            list of conditional transfer entropy calculated from nonlinear method using shuffled data.
        z_scores_linear  : `list`, optional
            list of z-scores calculated from linear method.
        z_scores_nonlinear  : `list`, optional
            list of z-scores calculated from nonlinear method.
        TE_list_lin_rev  : `list`, optional
            list of transfer entropy calculated from linear method in a reverse direction.
        TE_list_shuffle_lin_rev  : `list`, optional
            list of transfer entropy calculated from linear method using shuffled data in a reverse direction.
        TE_list_nonlin_rev  : `list`, optional
            list of transfer entropy calculated from nonlinear method in a reverse direction.
        TE_list_lin_con_rev  : `list`, optional
            list of conditional transfer entropy calculated from linear method in a reverse direction.
        TE_list_nonlin_con_rev  : `list`, optional
            list of conditional transfer entropy calculated from nonlinear method in a reverse direction.
        TE_list_shuffle_nonlin_rev  : `list`, optional
            list of transfer entropy calculated from nonlinear method using shuffled data in a reverse direction.
        TE_list_shuffle_lin_con_rev  : `list`, optional
            list of conditional transfer entropy calculated from linear method using shuffled data in a reverse direction.
        TE_list_shuffle_nonlin_con_rev  : `list`, optional
            list of conditional transfer entropy calculated from nonlinear method using shuffled data in a reverse direction.
        z_scores_lin_rev  : `list`, optional
            list of z-scores calculated from linear method in a reverse direction.
        z_scores_nonlin_rev  : `list`, optional
            list of z-scores calculated from nonlinear method in a reverse direction.
        signal : `bool`
            identify whether there is an error showing up in the regression model.
        """

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
        """ Update dataset.

        Parameters
        ----------
        new_dataset : `pandas.DataFrame`
            new dataset needed to be used

        """
        self.dataset = new_dataset

    def TE_calculation(self, jr, ir):
        """ Calculate transfer entropy.

        Parameters
        ----------
        jr : `list`
            list of residuals from the regression with more input variables
        ir : `list`
            list of residuals from the regression with less input variables

        Returns
        -------
        TE : `float`
            transfer entropy
        """

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
        """ Check if function sm.add_constant works.

        Parameters
        ----------
        Xpred_old : `numpy.ndarray`
            array before using sm.add_constant
        Xpred_new : `numpy.ndarray`
            array after using sm.add_constant
        Returns
        -------
        signal : `bool`
            error indicator
        """

        if Xpred_old.shape[1] == Xpred_new.shape[1]:
            self.signal = False
        return self.signal

    def linear_TE_XY(self, dataset, dependent_var="Y", splitting_percentage=0.7):
        """ Calculate TE by the linear method for two-variable cases.

        Parameters
        ----------
        dataset : `pandas.DataFrame`
            the dataset used for analysis.
        dependent_var : `str`, optional
            determine which one is the dependent variable.
        splitting_percentage : `float`, optional
            splitting percentage.

        Returns
        -------
        TE : `float`
            transfer entropy.
        """

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

    def linear_TE_XYZ(self, dataset, dependent_var="Y", splitting_percentage=0.7):
        """ Calculate TE and conditional TE by the linear method for three-variable cases.

        dataset : `pandas.DataFrame`
            the dataset used for analysis.
        dependent_var : `str`, optional
            determine which one is the dependent variable.
        splitting_percentage : `float`, optional
            splitting percentage.

        Returns
        -------
        TE : `float`
            transfer entropy.
        TE_con : `float`
            conditional transfer entropy.
        """

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

    def shuffle_series(self, DF):
        """ Shuffle the series.

        Parameters
        ----------
        DF : `pandas.DataFrame`
            the dataset needed to be shuffled.

        Returns
        -------
        shuffled_DF : `pandas.DataFrame`
            shuffled dataset.
        """
        shuffled_DF = DF.apply(np.random.permutation)

        return shuffled_DF

    def compute_z_scores_c(self, reverse=False):
        """ Calculate z-scores for two-variable cases.

        Parameters
        ----------
        reverse : `bool`, optional
            determine whether the analysis is for X->Y or Y->X.

        """
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
        """ Calculate z-scores for thee-variable cases.

        Parameters
        ----------
        reverse : `bool`, optional
            determine whether the analysis is for X->Y or Y->X.

        """
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
        """ Build up a MLP model.

        Parameters
        ----------
        X : `numpy.ndarray`
            data of independent variables.
        Y : `numpy.ndarray`
            data of the dependent variable.
        percentage : `float`
            splitting percentage.

        Returns
        -------
        resi : `list`
            list of residuals.
        """

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
        """ Calculate nonlinear TE.

        Parameters
        ----------
        X_jr : `numpy.ndarray`
            data of more independent variables.
        Y_jr : `numpy.ndarray`
            data of the dependent variable.
        X_ir : `numpy.ndarray`
            data of less independent variables.
        Y_ir : `numpy.ndarray`
            data of the dependent variable.
        percentage : `float`, optional
            splitting percentage.

        Returns
        -------
        TE : `float`
            transfer_entropy
        """

        joint_residuals = self.MLP(X_jr, Y_jr, percentage)
        independent_residuals = self.MLP(X_ir, Y_ir, percentage)
        TE = self.TE_calculation(joint_residuals, independent_residuals)

        return TE


class TE_cwp(TE):
    """ Child class for coupled wiener processes. """

    def __init__(self, T=1, N=200, alpha=0.5, lag=5, seed1=None, seed2=None):
        """ Initialize an instance for class TE_cwp.

        Parameters
        ----------
        T : `float`, optional
            time length of the generated series.
        N : `int`, optional
            time step of the generated series.
        alpha : `float`, optional
            coefficient with a range of [0,1]
        lag : `int`, optional
            time lag.
        seed1 : `int`, optional
            the number used to initialize the random number generator.
        seed2 : `int`, optional
            the number used to initialize the random number generator.
        """
        TE.__init__(self, lag)
        self.T = T
        self.N = N
        self.alpha = alpha
        self.seed1 = seed1
        self.seed2 = seed2
        self.dist = "cwp"

    def data_generation(self):
        """ Generate a coupled wiener process. """
        dataset = coupled_wiener_process(self.T, self.N, self.alpha, self.lag, self.seed1, self.seed2)
        self.dataset = dataset
        self.update_dataset(dataset)

    def experiment(self, reverse=False, splitting_percentage=0.7):
        """ Perform an experiment of calculating TE.

        Parameters
        ----------
        reverse : `bool`, optional
            decide whether X or Y is the dependant variable.
        splitting_percentage : `float`, optional
            splitting percentage.

        """
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
    """ Child class for ternary wiener processes. """

    def __init__(self, T=1, N=300, alpha=0.5, phi=0.5, beta=0.5, lag=5, seed1=None, seed2=None, seed3=None):
        """ Initialize an instance for class TE_twp.

        Parameters
        ----------
        T : `float`, optional
            time length of the generated series.
        N : `int`, optional
            time step of the generated series.
        alpha : `float`, optional
            coefficient with a range of [0,1]
        phi : `float`, optional
            coefficient with a range of [0,1]
        beta : `float`, optional
            coefficient with a range of [0,1]
        lag : `int`, optional
            time lag.
        seed1 : `int`, optional
            the number used to initialize the random number generator.
        seed2 : `int`, optional
            the number used to initialize the random number generator.
        seed3 : `int`, optional
            the number used to initialize the random number generator.

        """
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
        """ Generate a ternary wiener process. """
        dataset = ternary_wiener_process(self.T, self.N, self.alpha, self.phi, self.beta, self.lag, seed1=None, seed2=None, seed3=None)
        self.dataset = dataset
        self.update_dataset(dataset)

    def z_score_diff_TE(self):
        """ Calculate z-scores for TE difference. """
        diff_TE_linear = np.array(self.TE_list_linear_con) - np.array(self.TE_list_linear)

        mean = np.mean(np.array(self.TE_list_shuffle_linear_con)-np.array(self.TE_list_shuffle_linear))
        std = np.std(np.array(self.TE_list_shuffle_linear_con) - np.array(self.TE_list_shuffle_linear))

        z_scores_lin_diff = []

        for diff in diff_TE_linear:
            z_score = (diff - mean) / std
            z_scores_lin_diff.append(z_score)

        diff_TE_nonlinear = np.array(self.TE_list_nonlinear_con) - np.array(self.TE_list_nonlinear)

        mean = np.mean(np.array(self.TE_list_shuffle_nonlinear_con) - np.array(self.TE_list_shuffle_nonlinear))
        std = np.std(np.array(self.TE_list_shuffle_nonlinear_con) - np.array(self.TE_list_shuffle_nonlinear))

        z_scores_nonlin_diff = []

        for diff in diff_TE_nonlinear:
            z_score = (diff - mean) / std
            z_scores_nonlin_diff.append(z_score)

        mean_z_diff_lin = np.mean(z_scores_lin_diff)
        mean_z_diff_nonlin = np.mean(z_scores_nonlin_diff)

        return mean_z_diff_lin, mean_z_diff_nonlin

    def experiment(self, reverse=False, splitting_percentage=0.7):
        """ Perform an experiment of calculating TE.

        Parameters
        ----------
        reverse : `bool`, optional
            decide whether X or Y is the dependant variable.
        splitting_percentage : `float`, optional
            splitting percentage.

        """
        if reverse==False:
            dependent_var = "Y"
        else:
            dependent_var = "X"

        TE_linear, TE_linear_con = self.linear_TE_XYZ(self.dataset, dependent_var, splitting_percentage)

        X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset, reverse)
        TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

        X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(self.dataset, reverse)
        TE_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

        dataset_shuffle = self.shuffle_series(self.dataset)
        TE_shuffle_linear, TE_shuffle_linear_con = self.linear_TE_XYZ(dataset_shuffle, dependent_var, splitting_percentage)

        X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle, reverse)
        TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

        X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(dataset_shuffle, reverse)
        TE_shuffle_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

        if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False and TE_nonlinear_con != False and TE_linear_con != False:

            if reverse == False:
                self.TE_list_linear.append(TE_linear)
                self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                self.TE_list_nonlinear.append(TE_nonlinear)
                self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)

                self.TE_list_linear_con.append(TE_linear_con)
                self.TE_list_shuffle_linear_con.append(TE_shuffle_linear_con)
                self.TE_list_nonlinear_con.append(TE_nonlinear_con)
                self.TE_list_shuffle_nonlinear_con.append(TE_shuffle_nonlinear_con)

            else:
                self.TE_list_lin_rev.append(TE_linear)
                self.TE_list_shuffle_lin_rev.append(TE_shuffle_linear)
                self.TE_list_nonlin_rev.append(TE_nonlinear)
                self.TE_list_shuffle_nonlin_rev.append(TE_shuffle_nonlinear)

                self.TE_list_lin_con_rev.append(TE_linear_con)
                self.TE_list_shuffle_lin_con_rev.append(TE_shuffle_linear_con)
                self.TE_list_nonlin_con_rev.append(TE_nonlinear_con)
                self.TE_list_shuffle_nonlin_con_rev.append(TE_shuffle_nonlinear_con)


class TE_clm(TE):
    """ Child class for coupled logistic maps. """

    def __init__(self, X, Y, T=1, N=1000, alpha=0.4, epsilon=0.9, r=4):
        """ Initialize an instance for class TE_clm.

        Parameters
        ----------
        X : `float`
            initial value of X.
        Y : `float`
            initial value of Y.
        T : `float`, optional
            time length of the generated series.
        N : `int`, optional
            time step of the generated series.
        alpha : `float`, optional
            coefficient with a range of [0,1]
        epsilon : `float`, optional
            coefficient with a range of [0,1]
        r : `float`, optional
            positive influencing factor.
        """
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
        """ Generate a coupled logistic map. """
        dataset = coupled_logistic_map(self.X, self.Y, self.T, self.N, self.alpha, self.epsilon, self.r)
        dataset["Y_lagged"] = dataset["Y"].shift(periods=self.lag)
        dataset["X_lagged"] = dataset["X"].shift(periods=self.lag)
        dataset = dataset.dropna(axis=0, how='any')
        self.dataset = dataset
        self.update_dataset(dataset)

    def varying_XY(self):
        """ Vary the initial value of X and Y. """
        self.X = random.random()
        self.Y = random.random()

    def experiment(self, reverse=False, splitting_percentage=0.7):
        """ Perform an experiment of calculating TE.

        Parameters
        ----------
        reverse : `bool`, optional
            decide whether X or Y is the dependant variable.
        splitting_percentage : `float`, optional
            splitting percentage.

        """
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

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle, reverse)
            TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False:

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


class TE_tlm(TE):
    """ Child class for ternary logistic maps. """

    def __init__(self, X, Y, T=1, N=700, alpha=0.4, epsilon=0.9, r=4):
        """ Initialize an instance for class TE_tlm.

        Parameters
        ----------
        X : `float`
            initial value of X.
        Y : `float`
            initial value of Y.
        T : `float`, optional
            time length of the generated series.
        N : `int`, optional
            time step of the generated series.
        alpha : `float`, optional
            coefficient with a range of [0,1]
        epsilon : `float`, optional
            coefficient with a range of [0,1]
        r : `float`, optional
            positive influencing factor.
        """
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
        """ Generate a ternary logistic map. """
        dataset = ternary_logistic_map(self.X, self.Y, self.T, self.N, self.alpha, self.epsilon)
        dataset["Y_lagged"] = dataset["Y"].shift(periods=self.lag)
        dataset["X_lagged"] = dataset["X"].shift(periods=self.lag)
        dataset["Z_lagged"] = dataset["Z"].shift(periods=self.lag)
        dataset = dataset.dropna(axis=0, how='any')
        self.dataset = dataset
        self.update_dataset(dataset)

    def z_score_diff_TE(self):
        """ Calculate z-scores for TE difference. """
        diff_TE_linear = np.array(self.TE_list_linear_con) - np.array(self.TE_list_linear)

        mean = np.mean(np.array(self.TE_list_shuffle_linear_con) - np.array(self.TE_list_shuffle_linear))
        std = np.std(np.array(self.TE_list_shuffle_linear_con) - np.array(self.TE_list_shuffle_linear))

        z_scores_lin_diff = []

        for diff in diff_TE_linear:
            z_score = (diff - mean) / std
            z_scores_lin_diff.append(z_score)

        diff_TE_nonlinear = np.array(self.TE_list_nonlinear_con) - np.array(self.TE_list_nonlinear)

        mean = np.mean(np.array(self.TE_list_shuffle_nonlinear_con) - np.array(self.TE_list_shuffle_nonlinear))
        std = np.std(np.array(self.TE_list_shuffle_nonlinear_con) - np.array(self.TE_list_shuffle_nonlinear))

        z_scores_nonlin_diff = []

        for diff in diff_TE_nonlinear:
            z_score = (diff - mean) / std
            z_scores_nonlin_diff.append(z_score)

        mean_z_diff_lin = np.mean(z_scores_lin_diff)
        mean_z_diff_nonlin = np.mean(z_scores_nonlin_diff)

        return mean_z_diff_lin, mean_z_diff_nonlin

    def experiment(self, reverse=False, splitting_percentage=0.7):
        """ Perform an experiment of calculating TE.

        Parameters
        ----------
        reverse : `bool`, optional
            decide whether X or Y is the dependant variable.
        splitting_percentage : `float`, optional
            splitting percentage.

        """
        if reverse == False:
            dependent_var = "Y"
        else:
            dependent_var = "X"

        TE_linear, TE_linear_con = self.linear_TE_XYZ(self.dataset, dependent_var, splitting_percentage)

        if self.signal:
            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(self.dataset, reverse)
            TE_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(self.dataset, reverse)
            TE_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            dataset_shuffle = self.shuffle_series(self.dataset)
            TE_shuffle_linear, TE_shuffle_linear_con = self.linear_TE_XYZ(dataset_shuffle, dependent_var, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XY_Y(dataset_shuffle, reverse)
            TE_shuffle_nonlinear = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            X_jr, Y_jr, X_ir, Y_ir = data_for_MLP_XYZ_XY(dataset_shuffle, reverse)
            TE_shuffle_nonlinear_con = self.nonlinear_TE(X_jr, Y_jr, X_ir, Y_ir, splitting_percentage)

            if TE_linear != False and TE_shuffle_linear != False and TE_nonlinear != False and TE_shuffle_nonlinear != False and TE_nonlinear_con != False and TE_linear_con != False:
                if reverse == False:
                    self.TE_list_linear.append(TE_linear)
                    self.TE_list_shuffle_linear.append(TE_shuffle_linear)
                    self.TE_list_nonlinear.append(TE_nonlinear)
                    self.TE_list_shuffle_nonlinear.append(TE_shuffle_nonlinear)

                    self.TE_list_linear_con.append(TE_linear_con)
                    self.TE_list_shuffle_linear_con.append(TE_shuffle_linear_con)
                    self.TE_list_nonlinear_con.append(TE_nonlinear_con)
                    self.TE_list_shuffle_nonlinear_con.append(TE_shuffle_nonlinear_con)

                else:
                    self.TE_list_lin_rev.append(TE_linear)
                    self.TE_list_shuffle_lin_rev.append(TE_shuffle_linear)
                    self.TE_list_nonlin_rev.append(TE_nonlinear)
                    self.TE_list_shuffle_nonlin_rev.append(TE_shuffle_nonlinear)

                    self.TE_list_lin_con_rev.append(TE_linear_con)
                    self.TE_list_shuffle_lin_con_rev.append(TE_shuffle_linear_con)
                    self.TE_list_nonlin_con_rev.append(TE_nonlinear_con)
                    self.TE_list_shuffle_nonlin_con_rev.append(TE_shuffle_nonlinear_con)

        self.signal = True
