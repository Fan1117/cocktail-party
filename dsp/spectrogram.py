import numpy as np
import librosa

from mediaio.audio_io import AudioSignal


class MelConverter:

	N_FFT = 2048
	HOP_LENGTH = 128

	N_MEL_FREQS = 64
	FREQ_MIN_HZ = 300
	FREQ_MAX_HZ = 3400

	def __init__(self, sample_rate):
		self._SAMPLE_RATE = sample_rate

		self._MEL_FILTER = librosa.filters.mel(
			sr=self._SAMPLE_RATE,
			n_fft=MelConverter.N_FFT,
			n_mels=MelConverter.N_MEL_FREQS,
			fmin=MelConverter.FREQ_MIN_HZ,
			fmax=MelConverter.FREQ_MAX_HZ
		)

	def signal_to_mel_spectrogram(self, audio_signal):
		signal = audio_signal.get_data(channel_index=0)
		D = librosa.core.stft(signal, n_fft=MelConverter.N_FFT, hop_length=MelConverter.HOP_LENGTH)
		magnitude, phase = librosa.core.magphase(D)

		mel_spectrogram = np.dot(self._MEL_FILTER, magnitude)
		return librosa.amplitude_to_db(mel_spectrogram)

	def reconstruct_signal_from_mel_spectrogram(self, mel_spectrogram):
		mel_spectrogram = librosa.db_to_amplitude(mel_spectrogram)
		magnitude = np.dot(np.linalg.pinv(self._MEL_FILTER), mel_spectrogram)

		inverted_signal = griffin_lim(magnitude, MelConverter.N_FFT, MelConverter.HOP_LENGTH, n_iterations=10)

		inverted_audio_signal = AudioSignal(inverted_signal, self._SAMPLE_RATE)
		inverted_audio_signal.set_sample_type(np.int16, equalize=True)

		return inverted_audio_signal


def griffin_lim(magnitude, n_fft, hop_length, n_iterations):
	"""Iterative algorithm for phase retrival from a magnitude spectrogram."""
	phase_angle = np.pi * np.random.rand(*magnitude.shape)
	D = invert_magnitude_phase(magnitude, phase_angle)
	signal = librosa.istft(D, hop_length=hop_length)

	for i in range(n_iterations):
		D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
		_, phase = librosa.magphase(D)
		phase_angle = np.angle(phase)

		D = invert_magnitude_phase(magnitude, phase_angle)
		signal = librosa.istft(D, hop_length=hop_length)

	return signal


def invert_magnitude_phase(magnitude, phase_angle):
	phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
	return magnitude * phase
