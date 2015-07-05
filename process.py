from pylab import *
from scipy.io import wavfile
from ipdb import set_trace as pause
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


from scipy.signal import decimate

def process_rnn():
    """
    Should output a list of N_samplesx1 matrices ( 1 for each file )
    :return:
    """

    features = None
    labels = None
    file_paths = 'data/emodb'
    lag = 37
    frame_len = 1024
    frame_step = 1024
    n_filter = 40
    min_freq = 130
    max_freq = 6800
    n_fft = 1024
    fbank = None
    features = []
    labels = []
    for file_name in os.listdir(file_paths):

        if not file_name.endswith('.wav'):
            continue
        file_path = os.path.join(file_paths, file_name)
        freq, data = wavfile.read(file_path)
        # assuming 16 bit
        # Create features

        # try raw data first

        # transform in n_samplesx1 :
        # create labels
        sample_label = [1 for i in range(data.shape[0])]

        sample_label = np.asarray(sample_label)
        if file_name[5] == 'W':
            sample_label *= 0
        elif file_name[5] == 'L':
            sample_label *= 1
        elif file_name[5] == 'E':
            sample_label *= 2
        elif file_name[5] == 'A':
            sample_label *= 3
        elif file_name[5] == 'F':
            sample_label *= 4
        elif file_name[5] == 'T':
            sample_label *= 5
        elif file_name[5] == 'N':
            sample_label *= 6
        else:
            raise ValueError('Unknown label.')

        labels.append(sample_label)
        features.append(data)

    return features,labels

def process():
    features = []
    labels = []
    file_paths = 'data/emodb'
    lag = 37
    frame_len = 512
    frame_step = 512
    n_filter = 40
    min_freq = 130
    max_freq = 6800
    n_fft = 512
    fbank = None

    for file_name in os.listdir(file_paths):
        if not file_name.endswith('.wav'):
            continue
        file_path = os.path.join(file_paths, file_name)
        freq, data = wavfile.read(file_path)

        data, _ = signal_to_frames(data,frame_len,frame_step)

        data = apply_hamming(data)

        # Create features

        # try raw data first

        rfft_coeff = np.abs(np.fft.rfft(data, axis=1, n=n_fft))

        if fbank is None:
            fbank = tribank(n_filter=n_filter,
                            min_freq=min_freq,
                            max_freq=max_freq,
                            samp_rate=freq,
                            n_fft=n_fft)
            fbank = fbank.T

        filter_coeffs = np.dot(rfft_coeff, fbank).astype(np.float32)
        # normalize

        filter_coeffs = normalize_data(filter_coeffs,1)
        filter_coeffs = filter_coeffs.astype(np.float32)
        # iNTRODUCE LAG 
        
        # new_input_n = filter_coeffs.shape[0] // lag
        # left_over = filter_coeffs.shape[0] % lag
        # if left_over > 0:
        #     # add zeros to the end
        #     filter_coeffs = filter_coeffs[0:(-1 * left_over), :]
        # filter_shape = (new_input_n, lag * filter_coeffs.shape[1])
        # filter_coeffs = np.reshape(filter_coeffs, newshape=filter_shape)

        # create labels
        sample_label = [1 for i in range(data.shape[0])]

        sample_label = np.asarray(sample_label)
        if file_name[5] == 'W':
            sample_label *= 0
        elif file_name[5] == 'L':
            sample_label *= 1
        elif file_name[5] == 'E':
            sample_label *= 2
        elif file_name[5] == 'A':
            sample_label *= 3
        elif file_name[5] == 'F':
            sample_label *= 4
        elif file_name[5] == 'T':
            sample_label *= 5
        elif file_name[5] == 'N':
            sample_label *= 6
        else:
            raise ValueError('Unknown label.')

        labels.append(sample_label.astype(np.int32))

        features.append(filter_coeffs)

    return features,labels




def signal_to_frames(signal, frame_len, frame_step, win_func=None):
    """
    Divides the signal into several, possibly overlapping frames.

    :param signal:
    :param frame_len:
    :param frame_step:
    :return:
    """
    assert signal.ndim == 1

    signal_len = len(signal)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    num_frames = number_frames(signal_len, frame_len, frame_step)

    indices = indices_grid(frame_len, frame_step, num_frames)
    framed_signal = signal[indices]

    if win_func is not None:
        framed_signal = win_func(framed_signal)

    remain_signal = []
    # Add plus one to get first index
    # that is not in framed_signal
    max_idx = np.max(indices) + 1
    if max_idx <= signal_len - 1:
        remain_signal = np.r_[remain_signal, signal[max_idx:]]

    return framed_signal, remain_signal


def number_frames(signal_len, frame_len, frame_step):
    """
    Computes the number of frames for a given signal length.

    :param signal_len:
    :param frame_len:
    :param frame_step:
    :return:
    """
    frames = 1
    if signal_len > frame_len:
        temp = (1.0 * signal_len - frame_len)/frame_step
        frames += int(np.floor(temp))

    return frames


def indices_grid(frame_len, frame_step, num_frames):
    """
    Computes a grid of indices for possibly overlapping frames.
    :param frame_len:
    :param frame_step:
    :param num_frames:
    :return:
    """
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    return indices


def apply_hamming(frames, inv=False):
    """
    Computes either the hamming window or its inverse and applies
    it to a sequence of frames.

    :param frames: Frames with dimension num_frames x num_elements_per_frame
    :param inv: Indicates if the window should be inversed.
    :return:
    """
    M = frames.shape[1]
    win = np.hamming(M)**(-1) if inv else np.hamming(M)
    return frames * win



def concatenate(a,b):

    if a is None:
        a = b
    else:
        a = np.r_['0',a,b]

    return a





def tribank(n_filter=40, min_freq=300.0, max_freq=16000.0, samp_rate=32000.0, n_fft=1024):
    # TODO write test: can use example from http://practicalcryptography.com/miscellaneous/machine
    # -learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn1
    # sanity checks
    assert samp_rate/2.0 >= max_freq
    assert min_freq >= 0

    # compute filter bank bounds
    min_mel = hz2mel(min_freq)  # lower mel freq
    max_mel = hz2mel(max_freq)  # upper mel freq
    delta_mel = (max_mel - min_mel) / (n_filter + 1.0)
    scale = (n_fft + 1.0) / samp_rate

    bounds = [int(scale * mel2hz(min_mel + m * delta_mel)) for m in range(n_filter + 2)]

    assert len(bounds) == n_filter+2

    fbank = np.zeros(shape=(n_filter, n_fft/2+1))
    # iterate over filter
    for m in xrange(0, n_filter):
        # iterate over points
        for k in xrange(bounds[m], bounds[m+1]):
            fbank[m, k] = (float(k) - bounds[m])/(bounds[m+1]-bounds[m])
        for k in xrange(bounds[m+1], bounds[m+2]):
            fbank[m, k] = (bounds[m+2]-float(k))/(bounds[m+2]-bounds[m+1])

    return fbank


def hz2mel(freq):
    """
    Converts Hertz to Mel
    """
    return 2595. * np.log10(1+freq/700.0)


def mel2hz(mel):
    """
    Converts mel frequency to Hertz frequency.
    """
    return 700. * (10**(mel/2595.0)-1)


def normalize_data(features, frames_per_input_window,minmax=False):

    # gaussian normalization
    if not minmax:
        # reshape for normalization
        features_shape = features.shape
        norm_shape = (features.shape[0]*frames_per_input_window, features.shape[1]/frames_per_input_window)
        features = np.reshape(features, newshape=norm_shape)

        stdScaler = StandardScaler(with_mean=True, with_std=True)
        features = stdScaler.fit_transform(features) \
            .reshape((features.shape[0], features.shape[1]))

        # reshape to training shape
        features = np.reshape(features, newshape=features_shape)
    # min max normalization
    else:
        stdScaler = StandardScaler(with_mean=True, with_std=True)
        features = stdScaler.fit_transform(features)


    return features

