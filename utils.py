from __future__ import print_function
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.svm import SVC
import sys
import pandas

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ThresholdRandomForest(RandomForestClassifier):
	def __init__(self, n_estimators=50, threshold = 0.5, **kwargs):
		super().__init__(n_estimators, **kwargs)
		self.threshold = threshold

	def predict(self, X):
		return [1 if len(t) > 1 and t[1] >= self.threshold else 0 for t in self.predict_proba(X)]

class ThresholdAdaBoost(AdaBoostClassifier):
	def __init__(self, threshold = 0.5, **kwargs):
		super().__init__(**kwargs)
		self.threshold = threshold

	def predict(self, X):
		return [1 if len(t) > 1 and t[1] >= self.threshold else 0 for t in self.predict_proba(X)]

class ThresholdKNeighbors(KNeighborsClassifier):
	def __init__(self, threshold = 0.5, **kwargs):
		super().__init__(**kwargs)
		self.threshold = threshold

	def predict(self, X):
		return [1 if len(t) > 1 and t[1] >= self.threshold else 0 for t in self.predict_proba(X)]

class ThresholdSVC(SVC):
	def __init__(self, threshold = 0.5, **kwargs):
		super().__init__(**kwargs)
		self.threshold = threshold

	def predict(self, X):
		return [1 if len(t) > 1 and t[1] >= self.threshold else 0 for t in self.predict_proba(X)]

class ThresholdGridCV():
	def __init__(self, model, params, cv=10, metric = f1_score, verbose = 2, random_state = 42):
		self._model = model
		self._params = params
		if type(cv) == int:
			self._cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state = random_state)
		elif type(cv) == KFold or type(cv) == StratifiedKFold:
			self._cv = cv
		else:
			raise ValueError("cv must be an integer or a KFold object")

		self._metric = metric
		self._verbose = verbose

		self._models = []
		self._scores = []
		self._avg_scores = []


	def fit(self, X, y, optimize=False):
		self._models = []
		self._scores = []
		self._avg_scores = []
		self._thresholds = []
		self._scores_train = []

		if isinstance(X, pandas.core.series.Series) or isinstance(X, pandas.core.frame.DataFrame):
			if self._verbose >= 2:
				print("Converting X in numpy array")
			X = X.values

		if isinstance(y, pandas.core.series.Series) or isinstance(y, pandas.core.frame.DataFrame):
			if self._verbose >= 2:
				print("Converting y in numpy array")
			y = y.values

		for params in list(ParameterGrid(self._params)):
			if self._verbose >=1:
				print("Evaluating model with parameters: {0}".format(params))

			tmp_scores = []
			tmp_scores_train = []
			tmp_thresholds = []
			model = self._model(**params)

			fold = 1
			for tr_index, tst_index in self._cv.split(X, y):
				X_train, X_test = X[tr_index], X[tst_index]
				y_train, y_test = y[tr_index], y[tst_index]
				
				if self._verbose >= 2:
					print("Training fold #{0}".format(fold), end = '')
					fold += 1

				model = model.fit(X_train, y_train)

				if optimize:
					best_threshold, best_f1_train = optimize_for_threshold(model, X_train, y_train)
				else:
					best_threshold, best_f1_train = get_best_threshold(model, X_train, y_train)
					
				predictions = (get_probabilities(model, X_test) >= best_threshold)*1

				score = self._metric(predictions, y_test)

				if self._verbose >= 2:
					print("\t{0}".format(score))
					sys.stdout.flush()

				tmp_scores.append(score)
				tmp_thresholds.append(best_threshold)
				tmp_scores_train.append(best_f1_train)

			model = model.fit(X, y)

			self._models.append(model)
			self._scores.append(tmp_scores)
			self._avg_scores.append(sum(tmp_scores) / len(tmp_scores))
			self._thresholds.append(tmp_thresholds)
			self._scores_train.append(tmp_scores_train)

		if self._verbose >= 1:
			print("Evaluation finished")
			print("Average score values: {0}".format(self._avg_scores))
			sys.stdout.flush()


	def best_model(self):
		if len(self._models) == 0 or len(self._scores) == 0:
			raise Exception("Not fitted! Call fit before")

		max_index = self._avg_scores.index(max(self._avg_scores))
		return self._models[max_index]

	def get_best(self):
		"""
		Returns a tuple (model, thresholds, train_scores, val_scores)
		"""
		if len(self._models) == 0 or len(self._scores) == 0:
			raise Exception("Not fitted! Call fit before")

		max_index = self._avg_scores.index(max(self._avg_scores))

		return (self._models[max_index], self._thresholds[max_index], self._scores_train[max_index], self._scores[max_index])

	def best_score(self):
		if len(self._models) == 0 or len(self._scores) == 0:
			raise Exception("Not fitted! Call fit before")

		return max(self._avg_scores)


def get_probabilities(model, X):
	probabilities = model.predict_proba(X)

	return np.array([t[1] if len(t) >= 1 else 0.0 for t in probabilities])

def predict_with_threshold(model, X, threshold):
	probabs = get_probabilities(model, X)

	return 1 * (probabs >= threshold)

def plot_roc_curve(model, X, y):
	probabilities = get_probabilities(model, X)
	fpr, tpr, threshold = roc_curve(y, probabilities)
	roc_auc = auc(fpr, tpr)
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')

	plt.show()

def get_best_threshold(model, X, y, metric = f1_score, only_below_50 = True):
	probabilities = get_probabilities(model, X)

	fpr, tpr, threshold = roc_curve(y, probabilities)

	if only_below_50:
		threshold = threshold[threshold < 0.5]
	
	def get_score(threshold, probabilities, y_true):
		predicts = 1*(probabilities >= threshold)
		#predicts = [1 if p >= threshold else 0 for p in probabilities]
		return metric(predicts, y_true)

	scores = [get_score(t, probabilities, y) for t in threshold]
	max_index = scores.index(max(scores))

	#return threshold[max_index], scores[max_index]
	return threshold[max_index], scores[max_index]

def optimize_for_threshold(model, X, y, cv=10, random_state = 42, verbose=1):
	model_copy = clone(model) # Make a copy of the model

	kfolds = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True)
	try:
		X = X.values
		y = y.values
	except:
		pass

	thrs = []
	scores = []
	i = 1
	for tr_idx, tst_idx in kfolds.split(X, y):
		X_tr, y_tr = X[tr_idx], y[tr_idx]
		X_tst, y_tst = X[tst_idx], y[tst_idx]

		m = model_copy.fit(X_tr, y_tr)
		thr,score = get_best_threshold(m, X_tst, y_tst)
		if verbose >= 1:
		    print("Fold #{0} found thr: {1} and score: {2}".format(i, thr, score))
		    i+=1

		thrs.append(thr)
		scores.append(score)

	return np.mean(thrs), np.mean(scores)













