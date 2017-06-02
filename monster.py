from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import utils
from utils import ThresholdGridCV, get_best_threshold, predict_with_threshold
import numpy as np
import sys
import pickle
import pandas

class Kraken():
	def __init__(self, classifiers, params, random_state = 42):
		if len(classifiers) != len(params):
			raise ValueError('classifiers and params must have the same length')
		self._classifiers = classifiers
		self._params = params
		self._random_state = random_state

	def addmodel(self, model, params):
		self._classifiers.append(model)
		self._params.append(params)

	def fit(self, X, y, cv=10, verbose=2, metric=f1_score, test_holdout = True, optimize=False):
		self._best_models_in_cv = []

		if isinstance(X, pandas.core.series.Series) or isinstance(X, pandas.core.frame.DataFrame):
			if verbose >= 2:
				print("Converting X in numpy array")
			X = X.values

		if isinstance(y, pandas.core.series.Series) or isinstance(y, pandas.core.frame.DataFrame):
			if verbose >= 2:
				print("Converting y in numpy array")
			y = y.values

		if test_holdout:
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = self._random_state, stratify=y)
			print('Created test holdout. % positive class in test holdout = {0}'.format(sum(y_test)/len(y_test)))
		else:
			X_train, y_train = X, y

		num_model = 0
		for model, params in zip(self._classifiers, self._params):
			grid = utils.ThresholdGridCV(model, params, cv=cv, verbose = verbose, metric = metric)

			if verbose >= 1:
				print('\n\n========== Starting grid search for {0} =========='.format(model))

			grid.fit(X_train, y_train, optimize=optimize)

			best_model, thresholds, tr_scores, val_scores = grid.get_best()
			if verbose >= 1:
				print('========== Ending grid search for {0} =========='.format(model))
				print('\nBest model found is: {0}'.format(best_model))
				print('Best score training: {0} - threshold: {1}'.format(np.mean(tr_scores), np.mean(thresholds)))
				print('Best score validation: {0}'.format(grid.best_score()))

				sys.stdout.flush()

			model_summary = {
				'model' : best_model,
				'threshold_used' : np.mean(thresholds),
				'scores_cv' : val_scores,
				'scores_tr' : tr_scores,
				'thresholds' : thresholds,
				'avg_score' : grid.best_score()}

			self._best_models_in_cv.append(model_summary)

			f = open('grid_result_model_{0}'.format(num_model), 'wb')
			pickle.dump([grid, model_summary], f)
			f.close()

		# Evaluate on test
		best_overall = self._best_models_in_cv[0]
		for mod in self._best_models_in_cv:
			if mod['avg_score'] > best_overall['avg_score']:
				best_overall = mod

		if verbose >= 1:
			print('\n\n\nBest model overall is: {0}'.format(best_overall))
			print('Training on whole train set')

		best_overall['model'] = best_overall['model'].fit(X_train, y_train)

		if test_holdout:
			avg_thr = best_overall['threshold_used']
			final_score = metric(predict_with_threshold(best_overall['model'], X_test, avg_thr), y_test)
			print('\nFinal performance on test: {0}'.format(final_score))

		if verbose >= 2:
			print("Re training on whole dataset")
		self._best_overall = best_overall
		self._best_overall['model'] = self._best_overall['model'].fit(X, y)

	def predict(self, X):
		if isinstance(X, pandas.core.series.Series) or isinstance(X, pandas.core.frame.DataFrame):
			X = X.values

		return utils.predict_with_threshold(self._best_overall['model'], X, self._best_overall['threshold_used'])





























