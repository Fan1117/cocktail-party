from mediaio.audio_io import AudioSignal, AudioMixer
from dsp.spectrogram import MelConverter

speech_signal = AudioSignal.from_wav_file("/tmp/obama-orig.wav")
mel_converter = MelConverter(speech_signal.get_sample_rate(), n_fft=640, hop_length=160, n_mel_freqs=80, freq_min_hz=0, freq_max_hz=8000)

speech_spectrogram, phase = mel_converter.signal_to_mel_spectrogram(speech_signal, get_phase=True)

reconstructed_speech_signal = mel_converter.reconstruct_signal_from_mel_spectrogram(speech_spectrogram, phase)
reconstructed_speech_signal.save_to_wav_file("/tmp/reconstructed-orig.wav")
