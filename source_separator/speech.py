from external.audio_utils import *
from mediaio.audio_io import AudioSignal


class Speech:

	fft_size = 2048  # window size for the FFT
	step_size = fft_size / 16  # distance to slide along the window (in time)
	spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
	lowcut = 500  # Hz # Low cut for our butter bandpass filter
	highcut = 15000  # Hz # High cut for our butter bandpass filter

	# For mels
	n_mel_freq_components = 64  # number of mel frequency channels
	shorten_factor = 10  # how much should we compress the x-axis (time)
	start_freq = 300  # Hz # What frequency to start sampling our melS from
	end_freq = 8000  # Hz # What frequency to stop sampling our melS from

	mel_filter, mel_inversion_filter = create_mel_filter(
		fft_size=fft_size,
		n_freq_components=n_mel_freq_components,
		start_freq=start_freq,
		end_freq=end_freq
	)

	@staticmethod
	def signal_to_mel_spectogram(audio_signal):
		wav_spectrogram = pretty_spectrogram(
			audio_signal.get_data(channel_index=0).astype(np.float64),
			fft_size=Speech.fft_size,
			step_size=Speech.step_size,
			log=True,
			thresh=Speech.spec_thresh
		)

		return make_mel(wav_spectrogram, Speech.mel_filter, shorten_factor=Speech.shorten_factor)

	@staticmethod
	def reconstruct_signal_from_mel_spectogram(mel_spectogram, sample_rate):
		inverted_spectrogram = mel_to_spectrogram(
			mel_spectogram,
			Speech.mel_inversion_filter,
			spec_thresh=Speech.spec_thresh,
			shorten_factor=Speech.shorten_factor
		)

		inverted_signal = invert_pretty_spectrogram(
			np.transpose(inverted_spectrogram),
			fft_size=Speech.fft_size,
			step_size=Speech.step_size,
			log=True,
			n_iter=10
		)

		equalization_factor = np.iinfo(np.int16).max / np.abs(inverted_signal).max()
		reconstructed_signal = inverted_signal * equalization_factor

		return AudioSignal(reconstructed_signal.astype(np.int16), sample_rate)
