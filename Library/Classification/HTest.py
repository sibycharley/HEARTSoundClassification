import numpy as np
import pandas as pd
import sys
import pickle
import scipy.io.wavfile as wav

sys.path.append('../Feature_Extraction/')

from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn import metrics
from HModels import HEARTModels
from TestFeatures import TestFeat
from imblearn.metrics import sensitivity_score, specificity_score

class HEARTTest():

	#Define the initialization Code
	def __init__(self, Fcsv, tcsv, tdata):
		self.testfeat = Fcsv
		self.testcsv = tcsv
		self.testwav = tdata

	#Reading the Test csv file for getting the file names and labels
	def Read_Testcsv(self, feat):
		df = pd.read_csv(feat, header=None)
		return df	

	#Calculating Accuracy metrics based on maximum voting of the segments
	def CalcAcc(self, label):

		#Reading the input csv file
		df = pd.read_csv(self.testcsv)	
		ts_file, ts_label = df['File Name'].as_matrix(), df['Label'].as_matrix();
		tet = 2

		cnt = 0
		acc = 0
		ts_label[ts_label == -1] = 0
		pLab = np.zeros((len(ts_file)))

		for i in range(len(ts_file)):					#

			#Reading each of the input wavefiles
			xpath = self.testwav + ts_file[i] + ".wav"
			(Fs, inp) = wav.read(xpath)

			#Length of each input file
			flen = 3*Fs
			seglen = len(inp)/flen

			#Extracting the labels
			tmpl = label[cnt:cnt+seglen]
			pLab[i] = mode(tmpl)[0][0]

			if(pLab[i] == ts_label[i]):
				acc = acc + 1

			cnt = cnt + seglen	

		Se = sensitivity_score(ts_label, pLab)
		Sp = specificity_score(ts_label, pLab)	
		MAc = (Se+Sp)/2

#		print np.double(acc)/len(ts_file)
#		print np.sum(pLab == 1)
#		print np.sum(pLab == 0)
#		print np.sum(ts_label == 1)
#		print np.sum(ts_label == 0)	
		print "Final Sensitivity => " + str(Se)
		print "Final Specificity => " + str(Sp)
		print "Final Overall Accuracy => " + str(MAc)		

	#Feature Execution
	def HTest(self):

		#Reading the csv of feature vectors for Feature type 1
		feat = self.Read_Testcsv(self.testfeat).as_matrix()

		#Retrieving the Training Labels
		Narr_label = feat[:,27]
		feat = np.delete(feat, np.s_[16:27], axis=1)
		Narr_label[Narr_label == -1] = 0

		print np.sum(Narr_label == 0)
		print np.sum(Narr_label == 1)

		#Normalizing the Features
		nV = np.linalg.norm(feat, axis = 0, keepdims = True)
		nM = np.mean(feat, axis = 0, keepdims = True)
		nFeat = (feat - nM)/nV

		#Applying the PCA for reducing dimension to 6 
		pca_res = PCA(n_components = 6)
		nFeat_PCA = pca_res.fit_transform(nFeat)
		print pca_res.explained_variance_ratio_ 

		#Classification Algorithms Prediction on Test data

		Xtest = nFeat_PCA
		Ytest = Narr_label	

		#SVM
		with open('SVM_TrainModel.pkl', 'rb') as f:
			svm_learn = pickle.load(f)
		pred_label = svm_learn.predict(Xtest)
		print np.mean(pred_label == Ytest)
		print np.sum(pred_label == 1)
		print "************************"

		self.CalcAcc(pred_label)
		print "************************"
		print "************************"

		#Random Forest
		with open('RF_TrainModel.pkl', 'rb') as f:			
			rf_learn = pickle.load(f)
		pred_label = rf_learn.predict(Xtest)
		print np.mean(pred_label == Ytest)
		print np.sum(pred_label == 1)
		print "************************"

		self.CalcAcc(pred_label)
		print "************************"
		print "************************"

		#GBM
		with open('GBM_TrainModel.pkl', 'rb') as f:
			gbm_learn = pickle.load(f)
		pred_label = gbm_learn.predict(Xtest)
		print np.mean(pred_label == Ytest)
		print np.sum(pred_label == 1)
		print "************************"

		self.CalcAcc(pred_label)
		print "************************"
		print "************************"		
	
if __name__ == "__main__":

	#Define the file paths and directories
	Fcsv = "../../Data/Test_Feat_MFCC_zcr.csv"
	tcsv = "../../Data/Test.csv"
	tdata = "../../Data/TestData/"

	#Call the Testing constructor
	TestExt = HEARTTest(Fcsv, tcsv, tdata)

	#Call the Model testing Execution
	TestExt.HTest()	

	print "Hello"