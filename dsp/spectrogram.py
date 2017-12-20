import numpy as np
import librosa

from mediaio.audio_io import AudioSignal


class MelConverter:
	def __init__(self, sample_rate, n_fft=2048, hop_length=512, n_mel_freqs=128, freq_min_hz=0, freq_max_hz=None):
		self._SAMPLE_RATE = sample_rate
		self._N_FFT = n_fft
		self._HOP_LENGTH = hop_length

		self._N_MEL_FREQS = n_mel_freqs
		self._FREQ_MIN_HZ = freq_min_hz
		self._FREQ_MAX_HZ = freq_max_hz

		self._MEL_FILTER = librosa.filters.mel(
			sr=self._SAMPLE_RATE,
			n_fft=self._N_FFT,
			n_mels=self._N_MEL_FREQS,
			fmin=self._FREQ_MIN_HZ,
			fmax=self._FREQ_MAX_HZ
		)

		self._INVERSE_MEL_FILTER = np.linalg.pinv(self._MEL_FILTER)

	def signal_to_mel_spectrogram(self, audio_signal, get_phase=False):
		signal = audio_signal.get_data(channel_index=0)
		D = librosa.core.stft(signal, n_fft=self._N_FFT, hop_length=self._HOP_LENGTH)
		magnitude, phase = librosa.core.magphase(D)

		# used to get image of raw spectogram
		# return librosa.amplitude_to_db(magnitude)

		mel_spectrogram = np.dot(self._MEL_FILTER, magnitude)

		if get_phase:
			return librosa.amplitude_to_db(mel_spectrogram), phase
		else:
			return librosa.amplitude_to_db(mel_spectrogram)

	def reconstruct_signal_from_spectrogram(self, spectrogram, original_phase=None, peak=None, mel=True, db=True):
		if mel:
			magnitude = self.reconstruct_spectrogram_from_mel(spectrogram)
		else:
			if db:
				magnitude = librosa.db_to_amplitude(spectrogram)
			else:
				magnitude = spectrogram

		if original_phase is not None:
			inverted_signal = librosa.istft(magnitude * original_phase, hop_length=self._HOP_LENGTH)
		else:
			inverted_signal = griffin_lim(magnitude, self._N_FFT, self._HOP_LENGTH, n_iterations=10)

		inverted_audio_signal = AudioSignal(inverted_signal, self._SAMPLE_RATE)

		if peak is not None:
			inverted_audio_signal.peak_denormalize(peak)

		inverted_audio_signal.set_sample_type(np.int16)

		return inverted_audio_signal

	def reconstruct_spectrogram_from_mel(self, mel_spectrogram):
		mel_spectrogram = librosa.db_to_amplitude(mel_spectrogram)
		magnitude = np.dot(self._INVERSE_MEL_FILTER, mel_spectrogram)

		return magnitude

	def sectogram_to_mel(self, spectrogram):
		mel_spectrogram = np.dot(self._MEL_FILTER, spectrogram)
		log_mel = librosa.db_to_amplitude(mel_spectrogram)

		return log_mel

	def get_n_mel_freqs(self):
		return self._N_MEL_FREQS

	def get_hop_length(self):
		return self._HOP_LENGTH


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
