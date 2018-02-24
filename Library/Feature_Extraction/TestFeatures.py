import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

from Mtest import MelFreq
from scipy import signal

class TestFeat():

	#Define the initialization Code
	def __init__(self, tcsv, tdata, Fcsv):
		self.testcsv = tcsv
		self.testwav = tdata
		self.testfeat = Fcsv

	#Reading the Test csv file for getting the file names and labels
	def Read_Testcsv(self):
		df = pd.read_csv(self.testcsv)
		return df['File Name'], df['Label']
	
	#Reading the input wavfile and returning samples
	def Read_Testwav(self, fname):
		xpath = self.testwav + fname
		(Fs, inp) = wav.read(xpath)
		return inp, Fs

	#Write the Feature vectors onto csv
	def Write_Featvect(self, xinp):
		df = pd.DataFrame(xinp)
		df = df.transpose()
		df.to_csv(self.testfeat, mode='a', header=None, index = False)

	#Feature Execution
	def Fexecute(self):

		#Defining the MFCC class
		mfrq = MelFreq()

		#Reading the csv
		Narr_file, Narr_label = self.Read_Testcsv()
		tet = 1
		cnt = 0

		for i in range(len(Narr_file)):

			#Reading the wav
			inp, Fs = self.Read_Testwav(Narr_file[i] + ".wav")

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

				#Zero Crossing rate
				arr = inp[fcnt:fcnt+flen-1]
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
	tcsv = "../../Data/Test.csv"
	tdata = "../../Data/TestData/"
	Fcsv = "../../Data/Test_Feat_MFCC_zcr.csv"

	#Call the Feature Extraction constructor
	TFeat = TestFeat(tcsv, tdata, Fcsv)

	#Call the Feature Extraction Execution
	TFeat.Fexecute()	

	print "Hello"