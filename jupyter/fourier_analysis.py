import numpy as np
import scipy
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import librosa
import librosa.display
import IPython.display as display, Audio

import sys
sys.path.append('E:\Engineering\Signal Processing\MIR\FMP')
import libfmp.b

class Fourier_Analysis:
    def __init__(self, x):
        self.x = x.astype(np.complex128)
        
    # DFT
    def generate_matrix_dft(self, N, K):
        """
        Generates a DFT (distrete Fourier transform) matrix

        Parameters
        ----------
        N (int): Number of samples
        K (int): NUmber of frequency bins

        Returns
        -------
        dft_matrix (np.ndarray): DFT matrix

        """
        dft_matrix = np.zeros((K, N), dtype=np.complex128)
        
        for n in range(N):
            for k in range(K):
                dft_matrix[k, n] = np.exp(-2j*np.pi*k*n/N)
                
        return dft_matrix
    
    def generate_matrix_dft_inv(self, N, K):
        """
        Generates an IDFT (inverse distrete Fourier Transform) matrix

        Parameters
        ----------
        N (int): Number of samples
        K (int): Number of frequency bins

        Returns
        -------
        dft (np.ndarray): The IDFT matrix

        """
        dft = np.zeros((K, N), dtype=np.complex128)
        
        for n in range(N):
            for k in range(K):
                dft[k, n] = np.exp(2j*np.pi*k*n/N)/N
                
        return dft
    
    def dft(self):
        """
        Computes the distrete Fourier transform (DFT)

        Returns
        -------
        X (np.ndarray): Fourier transform of x

        """
        N = len(self.x)
        dft_matrix = self.generate_matrix_dft(N, N)
        
        return np.dot(dft_matrix, self.x)
    
    # FFT
    def __twiddle(self, N):
        """
        Generate the twiddle factors used in the computation of the fast Fourier transform (FFT)

        Parameters
        ----------
        N (int): Number of samples 

        Returns
        -------
        sigma (np.ndarray): The twiddle factors

        """
        k = np.arange(N // 2)
        sigma = np.exp(-2j*np.pi*k/N)
        
        return sigma
    
    def __fft(self, x):
        """
        Compute the fast Fourier Transform (FFT)

        Returns
        -------
        X (np.ndarray): Fourier transform of x
        
        """
        x = x.astype(np.complex128)
        N = len(x)
        log2N = np.log2(N)
        assert log2N == int(log2N), 'N must be a power of two!'
        X = np.zeros(N, dtype=np.complex128)
    
        if N == 1:
            return x
        else:
            this_range = np.arange(N)
            A = self.__fft(x[this_range % 2 == 0])
            B = self.__fft(x[this_range % 2 == 1])
            C = self.__twiddle(N) * B
            X[:N//2] = A + C
            X[N//2:] = A - C
            return X
    
    def fft(self):
        """
        Compute the fast Fourier Transform (FFT)

        Returns
        -------
        X (np.ndarray): Fourier transform of x

        """
        x = self.x
        
        return self.__fft(x)
    
    def windowed_ft(self, t, Fs, w_pos_sec, w_len, w_type, upper_y=1.0):
        N = len(self.x)
        w_pos = int(Fs * w_pos_sec)
        w = np.zeros(N)
        w[w_pos:w_pos + w_len] = scipy.signal.get_window(w_type, w_len)
        x = self.x * w
        
        plt.figure(figsize=(8, 2))
    
        plt.subplot(1, 2, 1)
        plt.plot(t, x, c='k')
        plt.plot(t, w, c='r')
        plt.xlim([min(t), max(t)])
        plt.ylim([-1.1, 1.1])
        plt.xlabel('Time (seconds)')
    
        plt.subplot(1, 2, 2)
        X = np.abs(np.fft.fft(x)) / N * 2
        freq = np.fft.fftfreq(N, d=1/Fs)
        X = X[:N//2]
        freq = freq[:N//2]
        plt.plot(freq, X, c='k')
        plt.xlim([0, 50])
        plt.ylim([0, upper_y])
        plt.xlabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()
        
    def stft_basic(self, w, H=8, only_positive_frequencies=False):
        """
        Compute a basic version of the distrete short-time Fourier transform (STFT)

        Parameters
        ----------
        w (np.ndarray): Window Function
        H (int): Hopsize (Default value = 8)
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible) (Default value = False)

        Returns
        -------
        X = (np.ndarray): The distrete short-time Fourier transform

        """
        N = len(w)
        L = len(self.x)
        M = np.floor((L - N)/H).astype(int) + 1
        X = np.zeros((N, M), dtype='complex')
        
        for m in range(M):
            x_win = self.x[m*H:m*H+N]*w
            X_win = np.fft.fft(x_win)
            X[:, m] = X_win
            
        if only_positive_frequencies:
            K = 1 + N//2
            X = X[0:K, :]
            
        return X
    
    def pad_and_plot(self, t, Fs, pad_len_sec, pad_mode):
        pad_len = int(pad_len_sec * Fs)
        
        t = np.concatenate((np.arange(-pad_len, 0) / Fs, t, 
                            np.arange(len(self.x), len(self.x) + pad_len) / Fs))
        x = np.pad(self.x, pad_len, pad_mode)
        N = len(x)
            
        plt.figure(figsize=(8, 1.5))
        ax1 = plt.subplot(1, 2, 1)
        plt.plot(t, x, c='k')
        #plt.xlim([t[0], t[-1]])
        plt.xlim([-1.0, 11.0])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
    
        ax2 = plt.subplot(1, 2, 2)
        X = np.abs(np.fft.fft(x)) / Fs
        freq = np.fft.fftfreq(N, d=1/Fs)
        X = X[:N//2]
        freq = freq[:N//2]
        plt.plot(freq, X, c='k')
        plt.xlim([0, 7])
        plt.ylim([0, 3])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        plt.show()
        
        return ax1, ax2
    
    def compute_stft(self, Fs, N, H, L, pad_mode='constant', center=True):    
        X = librosa.stft(self.x, n_fft=L, hop_length=H, win_length=N, 
                         window='hann', pad_mode=pad_mode, center=center)
        Y = np.log(1 + 100 * np.abs(X) ** 2)
        F_coef = librosa.fft_frequencies(sr=Fs, n_fft=L)
        T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H) 
        return Y, F_coef, T_coef
    
    def plot_stft(self, Y, Fs, N, H, time_offset=0, time_unit='frames', xlim=None, ylim=None, title='', xlabel='', color='hot'):
        time_samples = np.arange(Y.shape[1])
        
        if time_unit == 'sec':
            time_sec = np.arange(Y.shape[1])*(H/Fs) + time_offset
            extent = [time_sec[0] - H/(2*Fs), time_sec[-1] + H/(2*Fs), 0, Fs/2]
            xlabel = 'Time (seconds)'
        else:
            time_samples = np.arange(Y.shape[1])
            extent=[time_samples[0]-1/2, time_samples[-1]+1/2, 0, Fs/2]                     
            xlabel='Time (frames)'
            
        plt.imshow(Y, cmap=color, aspect='auto', origin='lower', extent=extent)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency (Hz)')
        plt.title(title)
        plt.colorbar()
        
    def plot_compute_spectrogram(self, Fs, N, H, L, color='gray_r'):
        Y, F_coef, T_coef = self.compute_stft(self.x, Fs, N, H, L)
        plt.imshow(Y, cmap=color, aspect='auto', origin='lower')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency (bins)')
        plt.title('L=%d' % L)
        plt.colorbar()
        
    def compute_plot_DFT_extended(self, t, Fs, L):
        N = len(self.x)
        pad_len = L - N
        t_tilde = np.concatenate((t, np.arange(len(self.x), len(self.x) + pad_len) / Fs))
        x_tilde = np.concatenate((self.x, np.zeros(pad_len)))
        Y = np.abs(np.fft.fft(x_tilde)) / Fs    
        Y = Y[:L//2]
        freq = np.arange(L//2)*Fs/L
        # freq = np.fft.fftfreq(L, d=1/Fs)
        # freq = freq[:L//2]
        plt.figure(figsize=(12, 2))
        
        ax1 = plt.subplot(1, 3, 1)
        plt.plot(t_tilde, x_tilde, c='k')
        plt.title('Signal ($N$=%d)' % N)
        plt.xlabel('Time (seconds)')
        plt.xlim([t[0], t[-1]])
        
        ax2 = plt.subplot(1, 3, 2)
        plt.plot(t_tilde, x_tilde, c='k')
        plt.title('Padded signal (of size $L$=%d)' % L)
        plt.xlabel('Time (seconds)')
        plt.xlim([t_tilde[0], t_tilde[-1]])    
        
        ax3 = plt.subplot(1, 3, 3)
        plt.plot(freq, Y, c='k')
        plt.title('Magnitude DFT of padded signal ($L$=%d)' % L)
        plt.xlabel('Frequency (Hz)')
        plt.xlim([freq[0], freq[-1]])
        plt.tight_layout()           
    
        return ax1, ax2, ax3
    
    def interpolate_plot_DFT(self, Y, N, Fs, F_coef, rho, int_method):
        F_coef_interpol = np.arange(F_coef[0], F_coef[-1], Fs/(rho*N))
        Y_interpol = interp1d(F_coef, Y, kind=int_method)(F_coef_interpol)
        plt.figure(figsize=(6, 2))
        plt.plot(F_coef_interpol, Y_interpol, c='k')
        plt.title(r'Magnitude DFT (interpolation: %s, $\rho$=%d)'%(int_method,rho))
        plt.xlabel('Frequency (Hz)')
        plt.xlim([F_coef[0], F_coef[-1]])
        plt.tight_layout()
        
    def stft_convention_fmp(self, Fs, N, H, pad_mode='constant', center=True, mag=False, gamma=0):
        """
        Compute the distrete short-time Fourier Transform (STFT)

        Parameters
        ----------
        Fs (scalar): Sampling Rate
        N (int): Window size
        H (int): Hopsize
        pad_mode (str): Padding strategy is used in librosa. The default is 'constant'.
        center (bool): Centric view as used in librosa. The default is True.
        mag (bool): Computes magnitude STFT if mag==True. The default is False.
        gamma (float): Constant for logarithmic compression (only applied when mag==True) The default is 0.

        Returns
        -------
        X (np.ndarray): Distrete (magnitude) short-time Fourier Transform

        """
        X = librosa.stft(self.x, n_fft=N, hop_length=H, win_length=N, window='hann', pad_mode=pad_mode, center=center)
        
        if mag:
            X = np.abs(X)**2
            if gamma > 0:
                X = np.log(1 + gamma*X)
                
        F_coef = librosa.fft_frequencies(sr=Fs, n_fft=N)
        T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
        
        return X, T_coef, F_coef
    
    def compute_f_coef_linear(self, N, Fs, rho=1):
        """
        Refines the frequency vector by factor of rho
        
        Parameters
        ----------
        N (int): Window size
        Fs (scalar): Sampling rate
        rho (int): Factor for refinement (Default value = 1)

        Returns
        -------
        F_coef_new (np.ndarray): Refined frequency vector

        """
        L = rho*N
        F_coef_new = np.arange(0, L//2+1)*Fs/L
        
        return F_coef_new
    
    def interpolate_freq_stft(self, Y, F_coef, F_coef_new):
        """
        Interpolation of STFT along frequency axis

        Parameters
        ----------
        Y (np.ndarray): Magnitude STFT
        F_coef (np.ndarray): Vector of frequency values
        F_coef_new (np.ndarray): Vector of new frequency values

        Returns
        -------
        Y_interpol (np.ndarray): Interpolated magnitude STFT

        """
        compute_Y_interpol = interp1d(F_coef, Y, kind='cubic', axis=0)
        Y_interpol = compute_Y_interpol(F_coef_new)
        
        return Y_interpol
    
    def plot_compute_spectrogram_physical(self, Fs, N, H, xlim, ylim, rho=1, color='gray_r'):
        Y, T_coef, F_coef = self.stft_convention_fmp(self.x, Fs, N, H, mag=True, gamma=100)
        F_coef_new = self.compute_f_coef_linear(N, Fs, rho=rho)
        Y_interpol = self.interpolate_freq_stft(Y, F_coef, F_coef_new)    
        extent=[T_coef[0], T_coef[-1], F_coef[0], F_coef[-1]]
        plt.imshow(Y_interpol, cmap=color, aspect='auto', origin='lower', extent=extent)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title(r'$\rho$=%d' % rho)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.colorbar()
        
    def compute_f_coef_log(self, R, F_min, F_max):
        """
        Adopts the frequency vector in a logarithmic fashion

        Parameters
        ----------
        R (scalar): Resolution (cents)
        F_min (float): Minimum Frequency
        F_max (float): Maximum Frequency (not included)

        Returns
        -------
        F_coef_log (np.ndarray): Refined frequency vector with values given in Hz
        F_coef_cents (np.ndarray): Refined frequency vector with values given in cents
        
        Note: F_min serves as reference (0 cents)

        """
        n_bins = np.ceil(1200*np.log2(F_max/F_min)/R).astype(int)
        F_coef_log = 2**(np.arange(0, n_bins)*R/1200)*F_min
        F_coef_cents = 1200*np.log2(F_coef_log/F_min)
        
        return F_coef_log, F_coef_cents
    
    def plot_sum_window(self, w, H, L, title='', figsize=(5, 1.5)):
        N = len(w)
        M = np.floor((L - N)/H).astype(int) + 1
        w_sum = np.zeros(L)
        
        plt.figure(figsize=figsize)
        
        for m in range(M):
            w_shifted = np.zeros(L)
            w_shifted[m*H:m*H + N] = w
            plt.plot(w_shifted, 'k')
            w_sum = w_sum + w_shifted
            
        plt.plot(w_sum, 'r', linewidth=3)
        plt.xlim([0, L-1])
        plt.ylim([0, 1.1*np.max(w_sum)])
        plt.title(title)
        plt.tight_layout()
        plt.show()
        return w_sum
    
    def istft_basic(self, X, w, H, L):
        """
        Compute the inverse of the basic distrete short-time Fourier Transform (ISTFT)

        Parameters
        ----------
        X (np.ndarray): The distrete short-time Fourier Transform
        w (np.ndarray): Window function
        H (int): Hopsize
        L (int): Length of time signal

        Returns
        -------
        x (np.ndarray): Time signal

        """
        N = len(w)
        M = X.shape[1]
        x_win_sum = np.zeros(L)
        w_sum = np.zeros(L)
        
        for m in range(M):
            x_win = np.fft.ifft(X[:, m])
            # Avoid imaginary values (due to floating point arithmetic)
            x_win = np.real(x_win)
            x_win_sum[m * H:m * H + N] = x_win_sum[m * H:m * H + N] + x_win
            w_shifted = np.zeros(L)
            w_shifted[m * H:m * H + N] = w
            w_sum = w_sum + w_shifted
        
        # Avoid division by zero
        w_sum[w_sum == 0] = np.finfo(np.float32).eps
        x_rec = x_win_sum / w_sum
        
        return x_rec, x_win_sum, w_sum
    
    def print_plot(self, x, x_rec):
        print('Number of samples of x:    ', x.shape[0])
        print('Number of samples of x_rec:', x_rec.shape[0])
        if x.shape[0] == x_rec.shape[0]:
            print('Signals x and x_inv agree:', np.allclose(x, x_rec))
            plt.figure(figsize=(6, 2))
            plt.plot(x-x_rec, color='red')
            plt.xlim([0, x.shape[0]])
            plt.title('Differences between x and x_rec')
            plt.xlabel('Time (samples)');
            plt.tight_layout()
            plt.show()
        else:
            print('Number of samples of x and x_rec does not agree.')
            
    def generate_function(self, Fs, dur=1):
        """
        Generate example function

        Parameters
        ----------
        Fs (scalar): Sampling rate
        dur (float): Duration (in seconds) of signal to be genereated. The default is 1.

        Returns
        -------
        x (np.ndarray): Signal
        t (np.ndarray): Time axis (in seconds)

        """
        N = int(Fs*dur)
        t = np.arange(N)/Fs
        
        x = 1*np.sin(2*np.pi*(2*t - 0))
        x += 0.5*np.sin(2*np.pi*(6*t - 0.1))
        x += 0.1*np.sin(2*np.pi*(20*t - 0.2))
        
        return x, t
    
    def sampling_equidistant(self, x_1, t_1, Fs_2, dur=None):
        """
        Equidistant sampling of interpolated signal

        Parameters
        ----------
        x_1 (np.ndarray): Signal to be interpolated and sampled
        t_1 (np.ndarray): Time axis (in seconds) of x_1
        Fs_2 (scalar): Sampling rate used for equidistant sampling
        dur (float): Duration (in seconds) of sampled signal (Default value = None)

        Returns
        -------
        x (np.ndarray): Sampled signal
        t (np.ndarray): Time axis (in seconds) of sampled signal

        """
        
        if dur is None:
            dur = len(t_1)*t_1[1]
            
        N = int(Fs_2*dur)
        t_2 = np.arange(N)/Fs_2
        x_2 = interp1d(t_1, x_1, kind='linear', fill_value='extrapolate')(t_2)
        
        return x_2, t_2
    
    def reconstruction_sinc(self, x, t, t_sinc):
        """
        Reconstruction from sampled signal sinc-functions

        Parameters
        ----------
        x (np.ndarray): Sampled signal
        t (np.ndarray): Equidistant discrete time axis (in seconds) of x
        t_sinc (np.ndarray): Equidistant discrete time axis (in seconds) of signal to be reconstructed

        Returns
        -------
        x_sinc (np.ndarray): Reconstructed signal having time axis t_sinc

        """
        Fs = 1/t[1]
        x_sinc = np.zeros(len(t_sinc))
        
        for n in range(0, len(t)):
            x_sinc += x[n]* np.sinc(Fs*t_sinc - n)
            
        return x_sinc
    
    def plot_signal_reconstructed(self, t_1, x_1, t_2, x_2, t_sinc, x_sinc):
        plt.figure(figsize=(8, 2.2))
        plt.plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
        plt.stem(t_2, x_2, linefmt='r:', markerfmt='r.', basefmt='None', label='Samples', use_line_collection=True)
        plt.plot(t_1, x_sinc, 'b', label='Reconstructed signal')
        plt.title(r'Sampling rate $F_\mathrm{s} = %.0f$'%(1/t_2[1]))
        plt.xlabel('Time (seconds)')
        plt.ylim([-1.5, 1.5])
        plt.xlim([t_1[0], t_1[-1]])
        plt.legend(loc='upper right', framealpha=1)
        plt.tight_layout()
        plt.show()
        
    def quantize_uniform(self, x, quant_min=-1.0, quant_max=1.0, quant_level=5):
        """
        Uniform quantization approach

        Parameters
        ----------
        quant_min (float): Minimum quantization level (Default value = -1.0)
        quant_max (float): Maximum quantization level (Default value = 1.0)
        quant_level (int): Number of quantization levels (Default value = 5)

        Returns
        -------
        x_quant (np.ndarray): Quantized signal

        """
        x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
        x_normalize[x_normalize > quant_level - 1] = quant_level - 1
        x_normalize[x_normalize < 0] = 0
        x_normalize_quant = np.around(x_normalize)
        x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
        
        return x_quant
    
    def plot_graph_quant_function(self, ax, quant_min=-1.0, quant_max=1.0, quant_level=256, mu=255.0, quant='uniform'):
        """
        Helper function for plotting a graph of quantization function and quantization error

        Parameters
        ----------
        ax (mpl.axes.Axes): Axis
        quant_min (float): Minimum quantization level (Default value = -1.0)
        quant_max (float): Maximum quantization level (Default value = 1.0)
        quant_level (int): Number of quantization levels (Default value = 256)
        mu (float): Encoding parameter (Default value = 255.0)
        quant (str): Type of quantization (Default value = 'uniform')

        Returns
        -------
        None.

        """
        # x = np.linspace(quant_min, quant_max, 1000)
        x = self.x
        if quant == 'uniform':
            x_quant = self.quantize_uniform(x, quant_min=quant_min, quant_max=quant_max, quant_level=quant_level)
            quant_stepsize = (quant_max - quant_min) / (quant_level-1)
            title = r'$\lambda = %d, \Delta=%0.2f$' % (quant_level, quant_stepsize)
        if quant == 'nonuniform':
            x_quant = self.quantize_nonuniform_mu(x, mu=mu, quant_level=quant_level)
            title = r'$\lambda = %d, \mu=%0.1f$' % (quant_level, mu)
        error = np.abs(x_quant - x)
        ax.plot(x, x, color='k', label='Original amplitude')
        ax.plot(x, x_quant, color='b', label='Quantized amplitude')
        ax.plot(x, error, 'r--', label='Quantization error')
        ax.set_title(title)
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Quantized amplitude/error')
        ax.set_xlim([quant_min, quant_max])
        ax.set_ylim([quant_min, quant_max])
        ax.grid('on')
        ax.legend()
        
    def plot_signal_quant(self, t, x_quant, figsize=(8, 2), xlim=None, ylim=None, title=''):
        """
        Helper function for plotting a signal and its quantized version

        Parameters
        ----------
        t: Time
        x_quant: Quantized signal
        figsize: Figure size (Default value = (8, 2))
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        title: Title of figure (Default value = '')

        Returns
        -------
        None.

        """
        x = self.x
        plt.figure(figsize=figsize)
        plt.plot(t, x, color='gray', linewidth=1.0, linestyle='-', label='Original Signal')
        plt.plot(t, x_quant, color='red', linewidth=2.0, linestyle='-', label='Quantized Signal')
        if xlim is None:
            plt.xlim([0, t[-1]])
        else:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.legend(loc='upper right', framealpha=1)
        plt.tight_layout()
        plt.show()
        
    def display_signal_quant(self, x, Fs, number_of_bits):
        quant_level = 2 ** number_of_bits
        x_quant = self.quantize_uniform(x, quant_min=-1, quant_max=1, quant_level=quant_level)    
        print('Signal after uniform quantization (%d bits) :'%number_of_bits, flush=True)
        display( Audio(x_quant, rate=Fs) )
        
        return x_quant
    
    def encoding_mu_law(self, v, mu=255.0):
        """
        mu-law encoding

        Parameters
        ----------
        v (float): Value between -1 and 1
        mu (float): Encoding parameter (Default value = 255.0)

        Returns
        -------
        v_encode (float): Encoded value

        """
        v_encode = np.sign(v)*(np.log(1.0 + mu*np.abs(v))/np.log(1.0 + mu))
        
        return v_encode
    
    def decoding_mu_law(self, v, mu=255.0):
        """
        mu-law decoding

        Parameters
        ----------
        v (float): Value between -1 and 1
        mu (float): Decoding parameter (Default value = 255.0)

        Returns
        -------
        v_decode (float): Decode value

        """
        v_decode = np.sign(v)*(1.0/mu)*((1.0 + mu)**np.abs(v) - 1.0)
        
        return v_decode
    
    def plot_mu_law(self, mu=255.0, figsize=(8.5, 4)):
        """
        Helper function for plottign a signal and its quantized version

        Parameters
        ----------
        mu (float): Dencoding parameter (Default value = 255.0)
        figsize (tuple): Figure size (Default value = (8.5, 2))

        Returns
        -------
        None.

        """
        values = np.linspace(-1, 1, 1000)
        values_encoded = self.encoding_mu_law(values, mu=mu)
        values_decoded = self.decoding_mu_law(values, mu=mu)
        
        plt.figure(figsize=figsize)
        ax = plt.subplot(1, 2, 1)
        ax.plot(values, values, color='k', label='Original Values')
        ax.plot(values, values_encoded, color='b', label='Encoded Values')
        ax.set_title(r'$\mu$-law Encoding with $\mu=%.0f$' % mu)
        ax.set_xlabel('$v$')
        ax.set_ylabel(r'$F_\mu(v)$')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.grid('on')
        ax.legend()
        
        ax = plt.subplot(1, 2, 2)
        ax.plot(values, values, color='k', label='Original values')
        ax.plot(values, values_decoded, color='b', label='Decoded values')
        ax.set_title(r'$\mu$-law decoding with $\mu=%.0f$' % mu)
        ax.set_xlabel('$v$')
        ax.set_ylabel(r'$F_\mu^{-1}(v)$')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.grid('on')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    def quantize_nonuniform_mu(self, mu=255.0, quant_level=256):
        """
        Nonuniform quantization approach using mu-encoding

        Parameters
        ----------
        mu (float): Encoding parameter (Default Value = 255.0)
        quant_level (int): Number of quantization levels (Default Value = 256)

        Returns
        -------
        x_quant (np.ndarray): Quantized Signal

        """
        x_en = self.encoding_mu_law(self.x, mu=mu)
        x_en_quant = self.quantize_uniform(x_en, quant_min=-1, quant_max=1, quant_level=quant_level)
        x_quant = self.decoding_mu_law(x_en_quant, mu=mu)
        
        return x_quant
    
    def compare_quant_signal(self, Fs, number_of_bits):
        quant_level = 2**number_of_bits
        mu = quant_level - 1
        x_qu = self.quantize_uniform(self.x, quant_min=-1, quant_max=1, quant_level=quant_level)
        x_qn = self.quantize_nonuniform(self.x, mu=mu, quant_level=quant_level)
        
        libfmp.b.audio_player_list([self.x, x_qu, x_qn], [Fs, Fs, Fs], width=160, columns=['Original (16 bits)', 'Uniform (%d bits)'%number_of_bits, 'Nonuniform (%d bits)'%number_of_bits])
        
    def plot_interference(self, x1, x2, t, figsize=(8, 2), xlim=None, ylim=None, title=''):
        """
        Helper function for plotting two signals and its superposition

        Parameters
        ----------
        x1: Signal 1
        x2: Signal 2
        t: Time
        figsize: figure size (Default value = (8, 2))
        xlim: x limits (Default value = None)
        ylim: y limits (Default value = None)
        title: figure title (Default value = '')

        Returns
        -------
        None.

        """
        plt.figure(figsize=figsize)
        plt.plot(t, x1, color='gray', linewidth=1.0, linestyle='-', label='x1')
        plt.plot(t, x2, color='cyan', linewidth=1.0, linestyle='-', label='x2')
        plt.plot(t, (x1 + x2), color='red', linewidth=1.0, linestyle='-', label='x1 + x2')
        
        if xlim is None:
            plt.xlim([0, t[-1]])
        else:
            plt.xlim(xlim)
            
        if ylim is not None:
            plt.ylim(ylim)
            
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()