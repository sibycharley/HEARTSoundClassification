import numpy as np
import pandas as pd
import pickle

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import VotingClassifier
from imblearn.metrics import sensitivity_score, specificity_score


class HEARTModels():

	#Define the initialization Code
	def __init__(self, nFeat_PCA, Narr_label):
		self.Xtrain = nFeat_PCA
		self.Ytrain = Narr_label

	#Custom Scoring Calculation
	def MAcc(self, actual, pred):

		Se = sensitivity_score(actual, pred)
		Sp = specificity_score(actual, pred)

		Acc = (Se+Sp)/2

		print "Frame wise Sensitivity => " + str(Se)
		print "Frame wise Specificity => " + str(Sp)

		return Acc
		
	#SVM Implemnetation
	def SVMModel(self):

		#Parameter Grid definition
		parameter_candidates = 	{
  									'C': [0.01, 0.1, 1], 
  									'gamma': [10, 1, 0.1, 0.01, 0.001]
  									#'kernel': ['linear', 'rbf']
								}

		#Applying the GridSearh with 5 fold Cross Validation for finding optimal parameters						
		svm_cross_model = GridSearchCV(estimator=svm.SVC(), scoring='accuracy', param_grid=parameter_candidates, cv=5, n_jobs=-1)						
		svm_cross_model.fit(self.Xtrain, self.Ytrain)

		print "******************************"
		print svm_cross_model.best_score_
		print svm_cross_model.best_params_
		print "******************************"		

		#Training the model with optiml parametrs on complete dataset
		svm_model = svm.SVC(kernel='rbf', 
							C=svm_cross_model.best_estimator_.C, 
							gamma=svm_cross_model.best_estimator_.gamma, probability=True)
		svm_model.fit(self.Xtrain, self.Ytrain)
		pred_label = svm_model.predict(self.Xtrain)
		print np.mean(pred_label == self.Ytrain)
		print np.sum(pred_label == 1)
		print "******************************"

		print "Frame wise Overall Accuracy => " + str(self.MAcc(self.Ytrain, pred_label))
		print "******************************"

		scores = cross_val_score(svm_model, self.Xtrain, self.Ytrain, cv=5, scoring='accuracy')
		print scores
		print "******************************"		

		return svm_model, pred_label	

	#Random Forest Implementation
	def RFModel(self):

		#Parameter Grid definition
		parameter_candidates = 	{
									'n_estimators': range(100,400,100),
									'max_features': [3, 4],
									'min_samples_split': range(15, 30, 3),
									'max_depth': range(5, 20, 5),
									'min_samples_leaf': range(2, 6, 2),		
  								}
		
  		rf_model = RandomForestClassifier()						

  		#Applying the GridSearh with 5 fold Cross Validation for finding optimal parameters						
		rf_cross_model = GridSearchCV(estimator=rf_model, param_grid=parameter_candidates, cv=5, n_jobs=-1, scoring='accuracy')						
		rf_cross_model.fit(self.Xtrain, self.Ytrain)									

		print "******************************"
		print rf_cross_model.best_score_
		print rf_cross_model.best_params_
		print "******************************"

		
		#Training the model with optiml parametrs on complete dataset
		rf_model = RandomForestClassifier(n_estimators=rf_cross_model.best_estimator_.n_estimators, 
										  max_features=rf_cross_model.best_estimator_.max_features, 
										  min_samples_split=rf_cross_model.best_estimator_.min_samples_split, 
										  max_depth=rf_cross_model.best_estimator_.max_depth,
										  min_samples_leaf=rf_cross_model.best_estimator_.min_samples_leaf)
		rf_model.fit(self.Xtrain, self.Ytrain)
		pred_label = rf_model.predict(self.Xtrain)
		print np.mean(pred_label == self.Ytrain)
		print np.sum(pred_label == 1)
		print "******************************"
		
		print "Frame wise Overall Accuracy => " + str(self.MAcc(self.Ytrain, pred_label))
		print "******************************"

#		scores = cross_val_score(rf_model, self.Xtrain, self.Ytrain, cv=5, scoring='accuracy')
#		print scores
#		print "******************************"

		return rf_model, pred_label


	#Gradient Boosting Machine(GBM) Implementation
	def GBMModel(self):

		#Parameter Grid definition
		parameter_candidates = 	{
									'learning_rate': [0.01, 0.001],
									'n_estimators': [100, 200, 300, 400],
									'subsample': [0.6, 0.7, 0.8, 0.9]		
  								}	

		gbm_model = GradientBoostingClassifier()

		#Applying the GridSearh with 5 fold Cross Validation for finding optimal parameters						
		gbm_cross_model = GridSearchCV(estimator=gbm_model, param_grid=parameter_candidates, cv=5, n_jobs=-1)						
		gbm_cross_model.fit(self.Xtrain, self.Ytrain)

		print "******************************"
		print gbm_cross_model.best_score_
		print gbm_cross_model.best_params_
		print "******************************"

		#Training the model with optiml parametrs on complete dataset
		gbm_model = GradientBoostingClassifier(learning_rate=gbm_cross_model.best_estimator_.learning_rate, 
											   n_estimators=gbm_cross_model.best_estimator_.n_estimators,  
										  	   subsample=gbm_cross_model.best_estimator_.subsample)
		gbm_model.fit(self.Xtrain, self.Ytrain)
		pred_label = gbm_model.predict(self.Xtrain)
		
		print np.mean(pred_label == self.Ytrain)
		print np.sum(pred_label == 1)
		print "******************************"

		print "Frame wise Overall Accuracy => " + str(self.MAcc(self.Ytrain, pred_label))
		print "******************************"

#		scores = cross_val_score(gbm_model, self.Xtrain, self.Ytrain, cv=5, scoring='accuracy')
#		print scores
#		print "******************************"

		return gbm_model, pred_label


	#Voting Classifier Model Implementation	
	def VotingModel(self):

		#SVM
		with open('SVM_TrainModel.pkl', 'rb') as f:
			svm_learn = pickle.load(f)
	
		#Random Forest
		with open('RF_TrainModel.pkl', 'rb') as f:
			rf_learn = pickle.load(f)

		#GBM
		with open('GBM_TrainModel_78.pkl', 'rb') as f:
			gbm_learn = pickle.load(f)

		#Training the model with instances of other models on complete dataset	
		vc_model = VotingClassifier(estimators=[('gbm', gbm_learn), ('rf', rf_learn)], voting='soft')
		vc_model.fit(self.Xtrain, self.Ytrain)
		pred_label = vc_model.predict(self.Xtrain)
		
		print np.mean(pred_label == self.Ytrain)
		print np.sum(pred_label == 1)
		print "******************************"

		print "Frame wise Overall Accuracy => " + str(self.MAcc(self.Ytrain, pred_label))
		print "******************************"

#		scores = cross_val_score(vc_model, self.Xtrain, self.Ytrain, cv=5, scoring='accuracy')
#		print scores
#		print "******************************"

		return vc_model, pred_label

