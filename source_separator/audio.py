import os

from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.models import Sequential


class SpeechSeparator:

	def __init__(self, train_cache_path):
		self._model = Sequential()

		self._model.add(Dense(units=1024, input_dim=64 * 100))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.25))

		self._model.add(Dense(units=1024))
		self._model.add(BatchNormalization())
		self._model.add(Activation("relu"))
		self._model.add(Dropout(0.25))

		self._model.add(Dense(units=64 * 100))
		self._model.compile(loss='mean_squared_error', optimizer='adam')

		self._train_cache_path = train_cache_path
		if os.path.exists(self._train_cache_path):
			self._model.load_weights(self._train_cache_path)

	def train(self, x, y):
		self._model.fit(x, y, batch_size=32, epochs=100, verbose=1)
		self._model.save_weights(self._train_cache_path)

	def evaluate(self, x, y):
		score = self._model.evaluate(x, y, verbose=1)
		return score

	def predict(self, x):
		y = self._model.predict(x)
		return y

