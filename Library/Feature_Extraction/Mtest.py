#Helper file for computing the MFCC features

from __future__ import division
import numpy
import decimal
import math
from scipy.fftpack import dct

class MelFreq():

    def mfcc(self, signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
             nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
             winfunc=lambda x:numpy.ones((x,))):
        """Compute MFCC features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param numcep: the number of cepstrum to return, default 13
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
        :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
        """
        feat,energy = self.fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
        feat = numpy.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
        feat = self.lifter(feat,ceplifter)
        if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
        return feat

    def fbank(self, signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
              winfunc=lambda x:numpy.ones((x,))):
        """Compute Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy, unwindowed)
        """
        highfreq= highfreq or samplerate/2
        signal = self.preemphasis(signal,preemph)
        frames = self.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
        pspec = self.powspec(frames,nfft)
        energy = numpy.sum(pspec,1) # this stores the total energy in each frame
        energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

        fb = self.get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
        feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
        feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

        return feat,energy

    def hz2mel(self, hz):
        """Convert a value in Hertz to Mels

        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * numpy.log10(1+hz/700.)

    def mel2hz(self, mel):
        """Convert a value in Mels to Hertz

        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700*(10**(mel/2595.0)-1)

    def get_filterbanks(self, nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq= highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = self.hz2mel(lowfreq)
        highmel = self.hz2mel(highfreq)
        melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = numpy.floor((nfft+1)*self.mel2hz(melpoints)/samplerate)

        fbank = numpy.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        return fbank

    def lifter(self, cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
        magnitude of the high frequency DCT coeffs.

        :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
        """
        if L > 0:
            nframes,ncoeff = numpy.shape(cepstra)
            n = numpy.arange(ncoeff)
            lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
            return lift*cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra

    def delta(self, feat, N):
        """Compute delta features from a feature vector sequence.

        :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        :param N: For each frame, calculate delta features based on preceding and following N frames
        :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
        """
        if N < 1:
            raise ValueError('N must be an integer >= 1')
        NUMFRAMES = len(feat)
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        delta_feat = numpy.empty_like(feat)
        padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
        for t in range(NUMFRAMES):
            delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        return delta_feat

    def preemphasis(self, signal, coeff=0.95):
        """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
        :returns: the filtered signal.
        """
        return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

    def powspec(self, frames, NFFT):
        """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

        :param frames: the array of frames. Each row is a frame.
        :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
        :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
        """
        return 1.0 / NFFT * numpy.square(self.magspec(frames, NFFT))   

    def magspec(self, frames, NFFT):
        """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

        :param frames: the array of frames. Each row is a frame.
        :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
        :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
        """
        if numpy.shape(frames)[1] > NFFT:
            logging.warn(
                'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
                numpy.shape(frames)[1], NFFT)
        complex_spec = numpy.fft.rfft(frames, NFFT)
        return numpy.absolute(complex_spec)    

    def round_half_up(self, number):
        return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
        
    def rolling_window(self, a, window, step=1):
        # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


    def framesig(self, sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
        """Frame a signal into overlapping frames.

        :param sig: the audio signal to frame.
        :param frame_len: length of each frame measured in samples.
        :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
        :returns: an array of frames. Size is NUMFRAMES by frame_len.
        """
        slen = len(sig)
        frame_len = int(self.round_half_up(frame_len))
        frame_step = int(self.round_half_up(frame_step))
        if slen <= frame_len:
            numframes = 1
        else:
            numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

        padlen = int((numframes - 1) * frame_step + frame_len)

        zeros = numpy.zeros((padlen - slen,))
        padsignal = numpy.concatenate((sig, zeros))
        if stride_trick:
            win = winfunc(frame_len)
            frames = self.rolling_window(padsignal, window=frame_len, step=frame_step)
        else:
            indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
                numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
            indices = numpy.array(indices, dtype=numpy.int32)
            frames = padsignal[indices]
            win = numpy.tile(winfunc(frame_len), (numframes, 1))

        return frames * win     