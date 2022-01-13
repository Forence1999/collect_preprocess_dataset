import os, sys
import numpy as np
import librosa
from numpy.fft import rfft, irfft
from lib.audiolib import audio_segmenter_4_numpy

EPS = np.finfo(np.float32).eps
REF_POWER = 1e-12


class audioFeatureExtractor(object):
    def __init__(self, fs, fft_len=None, num_gcc_bin=64, num_mel_bin=64, datatype='mic', num_channel=4):
        '''
        extract features for audio
        :param fs: sample frequency
        :param fft_len: parameter for rfft
        :param num_gcc_bin: number of gcc-phat features
        :param num_mel_bin: number of log-mel features
        :param datatype: type of audio( 'mic' or 'foa')
        :param num_channel: number of channels
        '''
        super(audioFeatureExtractor, self).__init__()
        
        self.fs = fs
        self.fft_len = fft_len
        self.num_mel_bin = num_mel_bin
        self.num_gcc_bin = num_gcc_bin
        if self.fft_len is not None:
            self.mel_weight = librosa.filters.mel(sr=self.fs, n_fft=self.fft_len, n_mels=self.num_mel_bin).T
        
        self.eps = np.finfo(float).eps
        self.num_channel = num_channel
        self.datatype = datatype
    
    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (int(x) - 1).bit_length()
    
    @staticmethod
    def _next_lower_power_of_2(x):
        return 2 ** ((int(x)).bit_length() - 1)
    
    def get_rfft_spectrogram(self, audio, fft_len=None):  # TODO don't support log_mel anymore
        audio = np.array(audio)
        if audio.ndim == 3:
            audio = audio[0]
        
        fft_spectra = []
        for i in range(len(audio)):
            temp_rfft = rfft(audio[i], n=self.fft_len)
            fft_spectra.append(temp_rfft)
        
        return np.array(fft_spectra)
    
    def get_log_mel(self, audio=None, rfft_spectra=None):  # TODO don't support log_mel anymore
        raise AssertionError('TODO don\'t support log_mel anymore')
        audio = np.array(audio)
        if audio.ndim == 3:
            audio = audio[0]
        if rfft_spectra is None:
            rfft_spectra = self.get_rfft_spectrogram(audio)
        
        mag_spectra = np.abs(rfft_spectra) ** 2
        mel_spectra = np.matmul(mag_spectra, self.mel_weight)
        log_mel_spectra = librosa.power_to_db(mel_spectra)
        
        return log_mel_spectra
    
    def _get_foa_intensity_vectors(self, linear_spectra):
        pass
        # IVx = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        # IVy = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        # IVz = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])
        #
        # normal = self.eps + (np.abs(linear_spectra[:, :, 0]) ** 2 + np.abs(linear_spectra[:, :, 1]) ** 2 + np.abs(
        #     linear_spectra[:, :, 2]) ** 2 + np.abs(linear_spectra[:, :, 3]) ** 2) / 2.
        # # normal = np.sqrt(IVx**2 + IVy**2 + IVz**2) + self.eps
        # IVx = np.dot(IVx / normal, self.mel_weight)
        # IVy = np.dot(IVy / normal, self.mel_weight)
        # IVz = np.dot(IVz / normal, self.mel_weight)
        #
        # # we are doing the following instead of simply concatenating to keep the processing similar to mel_spec and gcc
        # foa_iv = np.dstack((IVx, IVy, IVz))
        # foa_iv = foa_iv.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        # if np.isnan(foa_iv).any():
        #     print('Feature extraction is generating nan outputs')
        #     exit()
        # return foa_iv
    
    def get_gcc_phat(self, audio=None, rfft_spectra=None):
        audio = np.array(audio)
        if audio.ndim == 3:
            audio = audio[0]
        if rfft_spectra is None:
            rfft_spectra = self.get_rfft_spectrogram(audio)
        
        # gcc_channels = self.nCr(self.num_channel, 2)
        gcc_ls = []
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                R = np.conj(rfft_spectra[i]) * rfft_spectra[j]
                cc = irfft(np.exp(1.j * np.angle(R)))
                # cc = irfft(R / (np.abs(R) + self.eps))
                cc = np.concatenate((cc[-self.num_gcc_bin // 2:], cc[:self.num_gcc_bin // 2]))
                gcc_ls.append(cc)
        return np.array(gcc_ls)
    
    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self, audio):
        rfft_spectra = self.get_rfft_spectrogram(audio)
        
        if self.datatype is 'foa':
            pass
            # extract intensity vectors
            # foa_iv = self._get_foa_intensity_vectors(spect)
            # feat = np.concatenate((mel_spect, foa_iv), axis=-1)
        elif self.datatype is 'mic':
            gcc_phat = self.get_gcc_phat(rfft_spectra=rfft_spectra)
            log_mel = self.get_log_mel(rfft_spectra=rfft_spectra)
            
            return np.array([gcc_phat, log_mel])
        
        else:
            raise ValueError('Unknown dataset format {}'.format(self.datatype))
    
    def preprocess_features(self):
        pass
        # Setting up folders and filenames
        # spec_scaler = None
        #
        # # pre-processing starts
        #
        # spec_scaler = preprocessing.StandardScaler()
        # for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
        #     print('{}: {}'.format(file_cnt, file_name))
        #     feat_file = np.load(os.path.join(self._feat_dir, file_name))
        #     spec_scaler.partial_fit(feat_file)
        #     del feat_file
        # joblib.dump(
        #     spec_scaler,
        #     normalized_features_wts_file
        # )
        #
        # print('Normalizing feature files:')
        # print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        # for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
        #     print('{}: {}'.format(file_cnt, file_name))
        #     feat_file = np.load(os.path.join(self._feat_dir, file_name))
        #     feat_file = spec_scaler.transform(feat_file)
        #     np.save(
        #         os.path.join(self._feat_dir_norm, file_name),
        #         feat_file
        #     )
        #     del feat_file
        #
        # print('normalized files written to {}'.format(self._feat_dir_norm))
    
    def get_STFT(self, audio, clip_ms_length, overlap_ratio=0.5):
        '''
        :param audio:   channels * time_points
        :param clip_ms_length:
        :param overlap_ratio: between two adjacent audio clips
        :return:  (num_channel * 2) * num_audio_clips * frequency_bins
        for the first dimension: real imag; real imag; real imag; real imag;
        '''
        step_ms_size = clip_ms_length * (1 - overlap_ratio)
        audio = np.array(audio)
        if audio.ndim == 3:
            audio = audio[0]
        
        feature_ls = []
        for aud in audio:
            audio_seg = audio_segmenter_4_numpy(aud, fs=self.fs, segment_len=clip_ms_length / 1000.,
                                                stepsize=step_ms_size / 1000., window='hann', padding=False,
                                                pow_2=False)
            feature = self.get_rfft_spectrogram(audio_seg, fft_len=None)
            feature = np.asarray(feature)
            feature_ls.append(feature.real)
            feature_ls.append(feature.imag)
        return np.array(feature_ls)


if __name__ == '__main__':
    test_audio = np.random.rand(4, 4096)
    fe = audioFeatureExtractor(fs=16000)
    stft_feature = fe.get_STFT(audio=test_audio, clip_ms_length=64, overlap_ratio=0.5)
    print(stft_feature)
