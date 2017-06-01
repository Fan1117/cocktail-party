import os

from keras.layers import Dense, Convolution2D, MaxPooling2D, BatchNormalization, Activation, Dropout, LeakyReLU, Flatten, Merge
from keras.models import Sequential


class SpeechSeparator:

	def __init__(self, train_cache_path):
		audio_model = self._init_audio_model()
		video_model = self._init_video_model()

		self._model = Sequential()
		self._model.add(Merge([audio_model, video_model], mode='concat'))

		self._model.add(Dense(units=512))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.2))

		self._model.add(Dense(units=512))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.2))

		self._model.add(Dense(units=64 * 100))
		self._model.compile(loss='mean_squared_error', optimizer='adam')

		self._train_cache_path = train_cache_path
		if os.path.exists(self._train_cache_path):
			self._model.load_weights(self._train_cache_path)

	def _init_audio_model(self):
		model = Sequential()

		model.add(Dense(units=1024, input_dim=64 * 100))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(0.2))

		model.add(Dense(units=1024))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(0.2))

		return model

	def _init_video_model(self):
		model = Sequential()
		model.add(Convolution2D(32, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=(75, 128, 128)))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(32, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Convolution2D(32, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Convolution2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Convolution2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Convolution2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
		model.add(BatchNormalization())
		model.add(Activation("tanh"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))
		model.add(Flatten())

		return model

	def train(self, x_audio, x_video, y):
		self._model.fit([x_audio, x_video], y, batch_size=64, epochs=30, verbose=1)
		self._model.save_weights(self._train_cache_path)

	def evaluate(self, x_audio, x_video, y):
		score = self._model.evaluate([x_audio, x_video], y, verbose=1)
		return score

	def predict(self, x_audio, x_video):
		y = self._model.predict([x_audio, x_video])
		return y
