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
		model.add(Dropout(0.5))

		model.add(ZeroPadding3D(padding=(1, 2, 2), name='zero2'))
		model.add(Convolution3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2'))
		model.add(Dropout(0.5))

		model.add(ZeroPadding3D(padding=(1, 1, 1), name='zero3'))
		model.add(Convolution3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3'))
		model.add(Dropout(0.5))

		model.add(TimeDistributed(Flatten()))

		model.add(Dense(256, kernel_initializer='he_normal', name='dense0'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(256, kernel_initializer='he_normal', name='dense1'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Flatten())

		model.add(Dense(512, kernel_initializer='he_normal', name='dense2'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(512, kernel_initializer='he_normal', name='dense3'))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.2))

		model.add(Dense(audio_spectrogram_size, name='output'))

		model.summary()

		optimizer = optimizers.adam(lr=0.01, decay=1e-6)
		model.compile(loss="mean_squared_error", optimizer=optimizer)

		return VideoToSpeechNet(model)

	@staticmethod
	def load(model_cache_path, weights_cache_path):
		with open(model_cache_path, "r") as model_fd:
			model = model_from_json(model_fd.read())

		model.load_weights(weights_cache_path)

		return VideoToSpeechNet(model)

	def train(self, x, y):
		self._model.fit(x, y, batch_size=32, validation_split=0.05, epochs=120, verbose=1)

	def predict(self, x):
		y = self._model.predict(x)
		return y

	def dump(self, model_cache_path, weights_cache_path):
		with open(model_cache_path, "w") as model_fd:
			model_fd.write(self._model.to_json())

		self._model.save_weights(weights_cache_path)
