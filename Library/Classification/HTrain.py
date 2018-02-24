import numpy as np
import pandas as pd
import pickle
import scipy.io.wavfile as wav

from scipy.stats import mode
from sklearn.decomposition import PCA
from HModels import HEARTModels
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import sensitivity_score, specificity_score

class HEARTTrain():

	#Define the initialization Code
	def __init__(self, Fcsv, trcsv, tdata, tscsv):
		self.trainfeat = Fcsv
		self.traincsv = trcsv
		self.trainwav = tdata
		self.testcsv = tscsv

	#Reading the Train csv file for getting the file names and labels
	def Read_Traincsv(self, feat):
		df = pd.read_csv(feat, header=None)
		return df	

	#Calculating Accuracy metrics based on maximum voting of the segments
	def CalcAcc(self, label):

		#Reading the input csv file
		df = pd.read_csv(self.traincsv)	
		tr_file, tr_label = df['File Name'].as_matrix(), df['Label'].as_matrix();
		tet = 2

		cnt = 0
		acc = 0
		tr_label[tr_label == -1] = 0
		pLab = np.zeros((len(tr_file)))

		df = pd.read_csv(self.testcsv)	
		ts_file, ts_label = df['File Name'].as_matrix(), df['Label'].as_matrix();

		for i in range(len(tr_file)):					#

			#Checking whether the file is present in Test list
			if(ts_file[ts_file == tr_file[i]] == tr_file[i]):
				continue

			#Reading each of the input wavefiles
			xpath = self.trainwav + tr_file[i] + ".wav"
			(Fs, inp) = wav.read(xpath)

			#Length of each input file
			flen = 3*Fs
			seglen = len(inp)/flen

			#Extracting the labels
			tmpl = label[cnt:cnt+seglen]
			pLab[i] = mode(tmpl)[0][0]

			if(pLab[i] == tr_label[i]):
				acc = acc + 1

			cnt = cnt + seglen	

		Se = sensitivity_score(tr_label, pLab)
		Sp = specificity_score(tr_label, pLab)	
		MAc = (Se+Sp)/2

		print np.double(acc)/len(tr_file)
		print "Final Sensitivity => " + str(Se)
		print "Final Specificity => " + str(Sp)
		print "Final Overall Accuracy => " + str(MAc)

	#Feature Execution
	def HTrain(self):

		#Reading the csv of feature vectors for Feature type 1
		feat = self.Read_Traincsv(self.trainfeat).as_matrix()

		#Retrieving the Training Labels
		Narr_label = feat[:,27]
		feat = np.delete(feat, np.s_[16:27], axis=1)
		Narr_label[Narr_label == -1] = 0
		
		#Normalizing the Features
		nV = np.linalg.norm(feat, axis = 0, keepdims = True)
		nM = np.mean(feat, axis = 0, keepdims = True)
		nFeat = (feat - nM)/nV

		#Applying the PCA for reducing dimension to 6 
		pca_res = PCA(n_components = 6)
		nFeat_PCA = pca_res.fit_transform(nFeat)
		print pca_res.explained_variance_ratio_ 

		#Applying the Classification Algorithms
		HMod = HEARTModels(nFeat_PCA, Narr_label)

		#SVM
		svm_learn, pred_label = HMod.SVMModel()
		with open('SVM_TrainModel.pkl', 'wb') as f:
			pickle.dump(svm_learn, f)
		self.CalcAcc(pred_label)	

		#Random Forest
		rf_learn, pred_label = HMod.RFModel()
		with open('RF_TrainModel.pkl', 'wb') as f:
			pickle.dump(rf_learn, f)
		print self.CalcAcc(pred_label)	

		#GBM
		gbm_learn, pred_label = HMod.GBMModel()
		with open('GBM_TrainModel.pkl', 'wb') as f:
			pickle.dump(gbm_learn, f)
		self.CalcAcc(pred_label)	

if __name__ == "__main__":

	#Define the file paths and directories
	Fcsv = "../../Data/Train_Feat_MFCC_zcr.csv"
	trcsv = "../../Data/Train.csv"
	tdata = "../../Data/TrainData/"
	tscsv = "../../Data/Test.csv"

	#Call the Training constructor
	TrainExt = HEARTTrain(Fcsv, trcsv, tdata, tscsv)

	#Call the Model training Execution
	TrainExt.HTrain()	


	print "Hello"