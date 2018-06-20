import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from helpers import unison_shuffled_copies

from sklearn.model_selection import train_test_split



def safe_ln(x):
	return np.log(x+0.0001)



class BaseEnsemble(object):
	"""Base Object for Ensembles
	Mostly there to give the plotting utility to all it's children"""
    
	def plot_residuals(self,X_train,y_train,X_test,y_test):
		plt.scatter(self.predict(X_train).ravel(),self.predict(X_train).ravel()-y_train.ravel(),c='b',s=40,alpha=.5,label='train data prediction')
		plt.scatter(self.predict(X_test).ravel(),self.predict(X_test).ravel()-y_test.ravel(),c='g',s=40,alpha=.5,label='test data prediction')

		plt.hlines(y=0,xmin=0,xmax=50)
		plt.legend()
		plt.show()
	def scatterplot(self,X_test,X=None,y=None):
		if y is not None and X is not None:
			plt.scatter(X,y,s=20, edgecolor="black",
				c="darkorange", label="data")
		y_hat, std = self.predict(X_test,std=True)
		plt.plot(X_test,y_hat,label = 'predictive Mean')

		#plt.plot(X,std)
		var = y_hat + std
		var2 = y_hat - std
		assert(np.shape(var)==np.shape(y_hat))

		plt.fill_between(X_test.ravel(), y_hat.ravel(), var, alpha=.3, color='b',
						 label='uncertainty')

		plt.fill_between(X_test.ravel(), y_hat.ravel(), var2, alpha=.3, color='b')
		#plt.scatter(X_test,y_hat,s=20, edgecolor="black",
		#        c="darkorange", label="prediction")
		plt.xlabel("data")
		plt.ylabel("target")
		plt.title("Ensemble")
		plt.legend()
		plt.show()

	def mutli_dimenstional_scatterplot(self,X_test,y_test,X=None,y=None,figsize=(20,50)):
        
		y_hat,std = self.predict(X_test,std=True)

		#plt.rcParams["figure.figsize"] = (20,20)
		plt.figure(figsize=figsize)
		#plt.scatter(X[:,5],y)

		num_features = len(X_test.T)
		for i,feature in enumerate(X_test.T):
			#sort the arrays
			s = np.argsort(feature)
			var = y_hat[s]-std[s]
			var2 = y_hat[s] +std[s]


			plt.subplot(num_features,1,i+1)
			plt.plot(feature[s],y_hat[s],label = 'predictive Mean',)
			plt.fill_between(feature[s].ravel(),y_hat[s].ravel(),var,alpha=.3, color='b',label='uncertainty')
			plt.fill_between(feature[s].ravel(),y_hat[s].ravel(),var2,alpha=.3, color='b')
			plt.scatter(feature[s],y_test[s],label='data',s=20, edgecolor="black",
				c="darkorange")
			plt.xlabel("data")
			plt.ylabel("target")
			plt.title("Ensemble")
			plt.legend()     
		plt.show()

        
    #evaluation
	
	def nlpd(self,X,y):
		y_hat,std = self.predict(X,std=True)
		
		return -np.mean(1/2 * safe_ln(std) + (y_hat - y)**2/(std+0.0001))
		
    
    
	def coverage_probability(self,X, y):

		y_hat,std = self.predict(X,std=True) 
		#print(y_hat.shape,std.shape,y.shape)

		CP = 0
		for pred, s, target in zip(y_hat, std, y):
			#print(len(pred))
			#print(len(s))
			#print(len(target))
			if pred + s > target > pred - s:
				CP += 1
		return CP / len(y)
    
	def error_uncertainty_correlation(self,X,y):
		prediction,variance = self.predict(X,std=True)

		error = (prediction - y)**2
		correlation = scipy.stats.pearsonr(error.flatten(),variance.flatten())

		#np.correlate(error.flatten(),variance.flatten())
		return correlation

	def y_predicts_uncertainty(self,X,y):
		prediction = self.predict(X,std=False)

		correlation = scipy.stats.pearsonr(prediction.flatten(),y.flatten())
		return correlation


	def y_predicts_error(self,X,y):
		prediction = self.predict(X,std=False)

		correlation = scipy.stats.pearsonr(prediction.flatten(),y.flatten())
		return correlation


	def error_target_normalcy(self,X,y):
		scipy.stats.normaltest


		prediction,variance = self.predict(X,std=True)

		error = (prediction - y)**2
		normalcy =  scipy.stats.normaltest(error.flatten())

		#np.correlate(error.flatten(),variance.flatten())
		return normalcy

	def compute_rsme(self,X,y):
		y_hat = self.predict(X,False)
		return np.sqrt(np.mean((y_hat - y)**2))


    #eval meta
	def self_evaluate(self,X,y):
        
		rsme = self.compute_rsme(X,y)

		cov_prob = self.coverage_probability(X,y)
		#print('coverage Probability is: {}'.format(cov_prob))
		err_var_corr = self.error_uncertainty_correlation(X,y)[0]
		#print('correlation of error and uncertainty is: {}'.format(err_var_corr)) #0 is the coefficient
		y_uncertainty_pred = self.y_predicts_uncertainty(X,y)[0]
		#print('correlation of target value and uncertainty is: {}'.format(y_uncertainty_pred)) #0 is the coefficient
		y_predicts_error = self.y_predicts_error(X,y)[0]
		#print('correlation of target value and error is: {}'.format(y_uncertainty_pred)) #0 is the coefficient
		target_error_normalcy = self.error_target_normalcy(X,y)[0]
		#print('error-target normalcy is {}'.format(target_error_normalicy))
		nlpd = self.nlpd(X,y)

		return {'rsme':rsme,
				'coverage probability':cov_prob,
			   'correlation between error and variance':err_var_corr,
				'NLPD':nlpd,
			   #'predictive power of y on the uncertainty':y_uncertainty_pred,
			   #'predictive power of y on the error': y_predicts_error,
			   'error normalcy':target_error_normalcy}





        
    

class RegressionEnsemble(BaseEnsemble):
    def __init__(self,
            num_models=None,
            model_type=None,
            seed = None):
        self.num_models = num_models or 10
        self.model_type = model_type or DecisionTreeRegressor
        self.seed = seed or 42
        self.regressor_list = []
        
    def fit(self, X_train,y_train):
        for i in range(self.num_models):
            try:
                new_regressor = self.model_type(random_state=self.seed + i)#random_state=self.seed+i)
            except:
                new_regressor = self.model_type()#random_state=self.seed+i)

                
            new_regressor.fit(X_train,y_train)
            self.regressor_list.append(new_regressor)
        return 'ensemble of {} {}s is hired and at the ready'.format(self.num_models,self.model_type.__name__)
    
            
    def predict(self,y_test,std=False):
        prediction_list = []
        for regressor in self.regressor_list:
            prediction_list.append(regressor.predict(y_test))
            
        predictive_means = np.mean(prediction_list,0)
        if not std:
            return predictive_means
        
        predictive_stds = np.std(prediction_list,0)
        return predictive_means, predictive_stds
            
        
    
class SubspaceEnsemble(RegressionEnsemble,BaseEnsemble):
    def __init__(self,
                num_models=None,
                model_type = None,
                seed=None,
                num_drop_dimensions=None,):
    
        super().__init__(num_models=num_models,
            model_type=model_type,
            seed = seed)
        self.num_drop_dimensions = num_drop_dimensions or 1
        
    def fit(self,X_train,y_train):
        
        for i in range(self.num_models):
            idx = np.random.choice(X_train.shape[0], X_train.shape[1]-self.num_drop_dimensions, replace=False)

            X_new = X_train[idx]
            y_new = y_train[idx]
            
            new_regressor = self.model_type()
            new_regressor.fit(X_new,y_new)
            self.regressor_list.append(new_regressor)




    
class BootstrapEnsemble(RegressionEnsemble, BaseEnsemble):
    """essentially a regression ensemble, except during the fitting part,
    sub-datasets are created
    Currently still shuffles :/
    Currently no putting data back into the drawer :/"""
    def __init__(self,
                 num_models=None,
                model_type=None,
                seed = None,
                keep_p = None):
        
        super().__init__(num_models=num_models,
            model_type=model_type,
            seed = seed)

        self.keep_p = keep_p or 0.7
        
    def fit(self,X_train,y_train):
        #print(X_train.size,y_train.size)
        for i in range(self.num_models):
            new_regressor = self.model_type()
            X_new, throwaway1, y_new ,throwaway2 = train_test_split(X_train, y_train, test_size=self.keep_p, random_state=42+i,shuffle=True)
            new_regressor.fit(X_new,y_new)
            self.regressor_list.append(new_regressor)
        return 'ensemble of {} {}s is hired and at the ready'.format(self.num_models,self.model_type.__name__)
 


class ShuffleEnsemble(BootstrapEnsemble, BaseEnsemble):
    """Essentially a Bootstrapensemble with keep-probability of 1, 
    so the data only get's shuffled differently for each model"""
    def __init__(self,
                 num_models=None,
                model_type=None,
                seed = None):
        
        super().__init__(num_models=num_models,
            model_type=model_type,
            seed = seed,
            keep_p = 1) #only change: no data is thrown away
        

    
    
class MixedRegressionEnsemble(BaseEnsemble):
    def __init__(self,
                models = []):
        self.models = models or [DecisionTreeRegressor(),LinearRegression()]
        #self.model_type.__name__
        
    def fit(self,X_train,y_train):
        for model in self.models:
            model.fit(X_train,y_train)
            
    def predict(self,X_test,std=False):
        prediction_list = []
        for model in self.models:
            prediction_list.append(model.predict(X_test))
        predictive_means = np.mean(prediction_list,0)
        if not std:
            return predictive_means
        
        predictive_stds = np.std(prediction_list,0)
        return predictive_means, predictive_stds
    
    
