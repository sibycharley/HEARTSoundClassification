import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

from Mtest import MelFreq
from scipy import signal

class FeatExt():

	#Define the initialization Code
	def __init__(self, trcsv, tdata, tscsv, Fcsv):
		self.traincsv = trcsv
		self.trainwav = tdata
		self.testcsv = tscsv
		self.trainfeat = Fcsv

	#Reading the Train csv file for getting the file names and labels
	def Read_Traincsv(self):
		df = pd.read_csv(self.traincsv)
		return df['File Name'], df['Label']
	
	#Reading the input wavfile and returning samples
	def Read_Trainwav(self, fname):
		xpath = self.trainwav + fname
		(Fs, inp) = wav.read(xpath)
		return inp, Fs

	#Write the Feature vectors onto csv
	def Write_Featvect(self, xinp):
		df = pd.DataFrame(xinp)
		df = df.transpose()
		df.to_csv(self.trainfeat, mode='a', header=None, index = False)
		
	#Feature Execution
	def Fexecute(self):

		#Defining the MFCC class
		mfrq = MelFreq()

		#Reading the csv
		Narr_file, Narr_label = self.Read_Traincsv()
		tet = 1

		df = pd.read_csv(self.testcsv)	
		ts_file, ts_label = df['File Name'].as_matrix(), df['Label'].as_matrix();
		cnt = 0

		for i in range(len(Narr_file)):		#tet

			#Checking whether the file is present in Test list
			if(ts_file[ts_file == Narr_file[i]] == Narr_file[i]):
				continue

			#Reading the wav
			inp, Fs = self.Read_Trainwav(Narr_file[i] + ".wav")

			#Framing into 3 second frames
			flen = (3*Fs)
			nframe = np.int(len(inp) / flen)
						
			##################################################################################
			fcnt = 1
			Fmfcc = 0
			Ener = 0	
			
			#Collecting the Feature Vector 1
			for j in range(nframe):
			
				#Invoking the MFCC
				mTmp = mfrq.mfcc(inp[fcnt:fcnt+flen-1], Fs)	

				#Finding the MFCC statistics
				mTmp_avg = np.mean(mTmp, 0)
				mTmp_std = np.std(mTmp, 0)	

				#Zero Crossing Rate
				arr = inp[fcnt:fcnt+flen-1]/np.max(inp[fcnt:fcnt+flen-1])
				ain = arr.tolist()
				zcr = (np.diff(np.sign(ain)) != 0).sum()

				#Accumulating the MFCC coefficients and Energy for each frame
				if(j == 0):
					Fmfcc = np.hstack((zcr, mTmp_avg))	
					Fmfcc = np.append(Fmfcc, mTmp_std)
					Fmfcc = np.append(Fmfcc, Narr_label[i])
				else:
					stack = np.hstack((zcr, mTmp_avg))
					stack = np.append(stack, mTmp_std)
					stack = np.append(stack, Narr_label[i])
					Fmfcc = np.vstack((Fmfcc, stack))	

				#Updating the buffer counter
				fcnt = fcnt + flen
			##################################################################################	

			MFCCVet = np.transpose(Fmfcc)

			#Write the vectors on to csv file
			self.Write_Featvect(MFCCVet)

if __name__ == "__main__":

	#Define the file paths and directories
	trcsv = "../../Data/Train.csv"
	tscsv = "../../Data/Test.csv"
	tdata = "../../Data/TrainData/"
	Fcsv = "../../Data/Train_Feat_MFCC_zcr.csv"

	#Call the Feature Extraction constructor
	FExt = FeatExt(trcsv, tdata, tscsv, Fcsv)

	#Call the Feature Extraction Execution
	FExt.Fexecute()	

	print "Hello"