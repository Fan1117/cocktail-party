from keras import optimizers
from keras.layers import Dense, Convolution3D, MaxPooling3D, ZeroPadding3D, Dropout, Flatten, BatchNormalization, LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json


class VideoToSpeechNet:

	def __init__(self, model):
		self._model = model

	@staticmethod
	def build(video_shape, audio_spectrogram_size):
		model = Sequential()

		model.add(ZeroPadding3D(padding=(1, 2, 2), name='zero1', input_shape=video_shape))
		model.add(Convolution3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1'))
		model.add(Dropout(0.25))

		model.add(ZeroPadding3D(padding=(1, 2, 2), name='zero2'))
		model.add(Convolution3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2'))
		model.add(Dropout(0.25))

		model.add(ZeroPadding3D(padding=(1, 1, 1), name='zero3'))
		model.add(Convolution3D(128, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3'))
		model.add(Dropout(0.25))

		model.add(TimeDistributed(Flatten(), name='time'))

		model.add(Dense(1024, kernel_initializer='he_normal', name='dense1'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(1024, kernel_initializer='he_normal', name='dense2'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Flatten())

		model.add(Dense(2048, kernel_initializer='he_normal', name='dense3'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(2048, kernel_initializer='he_normal', name='dense4'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

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
		first_tuned_layer_index = self._get_layer_names().index("time")

		for layer in self._model.layers[:first_tuned_layer_index]:
			layer.trainable = False

		self._model.summary()

		self.train(x, y, learning_rate=0.001, epochs=50)

	def predict(self, x):
		y = self._model.predict(x)
		return y

	def dump(self, model_cache_path, weights_cache_path):
		with open(model_cache_path, "w") as model_fd:
			model_fd.write(self._model.to_json())

		self._model.save_weights(weights_cache_path)

	def _get_layer_names(self):
		return [layer.name for layer in self._model.layers]
