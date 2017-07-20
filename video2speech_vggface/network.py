from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, LeakyReLU
from keras.models import Sequential, model_from_json


class VideoToSpeechNet:

	def __init__(self, model):
		self._model = model

	@staticmethod
	def build(features_shape, audio_spectrogram_size):
		model = Sequential()

		model.add(Flatten(input_shape=features_shape))
		model.add(BatchNormalization())

		model.add(Dense(1024, kernel_initializer='he_normal', name='dense1'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.25))

		model.add(Dense(1024, kernel_initializer='he_normal', name='dense2'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.25))

		model.add(Dense(1024, kernel_initializer='he_normal', name='dense3'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.25))

		model.add(Dense(audio_spectrogram_size, name='output'))

		model.summary()

		return VideoToSpeechNet(model)

	@staticmethod
	def load(model_cache_path, weights_cache_path):
		with open(model_cache_path, "r") as model_fd:
			model = model_from_json(model_fd.read())

		model.load_weights(weights_cache_path)

		return VideoToSpeechNet(model)

	def train(self, x, y, learning_rate=0.01, epochs=200):
		optimizer = optimizers.adam(lr=learning_rate, decay=1e-6)
		self._model.compile(loss="mean_squared_error", optimizer=optimizer)

		self._model.fit(x, y, batch_size=32, validation_split=0.05, epochs=epochs, verbose=1)

	def fine_tune(self, x, y):
		self.train(x, y, epochs=50)

	def predict(self, x):
		y = self._model.predict(x)
		return y

	def dump(self, model_cache_path, weights_cache_path):
		with open(model_cache_path, "w") as model_fd:
			model_fd.write(self._model.to_json())

		self._model.save_weights(weights_cache_path)

