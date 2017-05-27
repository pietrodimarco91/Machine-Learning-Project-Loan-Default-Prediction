import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ThresholdRandomForest(RandomForestClassifier):
	def __init__(self, n_estimators=50, threshold = 0.5, **kwargs):
		super().__init__(n_estimators, **kwargs)
		self.threshold = threshold

	def predict(self, X):
		print('hey')
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
	def __init__(self, model, params, cv=10, metric = accuracy_score, verbose = 0, random_state = 42):
		self._model = model
		self._params = params
		if type(cv) == int:
			self._cv = KFold(n_splits=cv, shuffle=True, random_state = random_state)
		elif type(cv) == KFold:
			self._cv = cv
		else:
			raise ValueError("cv must be an integer or a KFold object")

		self._metric = metric
		self._verbose = verbose

		self._models = []
		self._scores = []
		self._avg_scores = []


	def fit(self, X, y):
		self._models = []
		self._scores = []
		self._avg_scores = []

		if isinstance(X, pandas.core.frame.DataFrame):
			if self._verbose >= 2:
				print("Converting data in numpy array")
			X = X.values

		for params in list(ParameterGrid(self._params)):
			if self._verbose >=1:
				print("Evaluating model with parameters: {0}".format(params))

			tmp_scores = []
			model = self._model(**params)

			fold = 1
			for tr_index, tst_index in self._cv.split(X):
				X_train, X_test = X[tr_index], X[tst_index]
				y_train, y_test = y[tr_index], y[tst_index]
				
				if self._verbose >= 2:
					print("Training fold #{0}".format(fold), end = '')
					fold += 1

				model = model.fit(X_train, y_train)
				score = self._metric(model.predict(X_test), y_test)

				if self._verbose >= 2:
					print("\t{0}".format(score))

				tmp_scores.append(score)

			model = model.fit(X, y)


			self._models.append(model)
			self._scores.append(tmp_scores)
			self._avg_scores.append(sum(tmp_scores) / len(tmp_scores))

		if self._verbose >= 1:
			print("Evaluation finished")
			print("Average score values: {0}".format(self._avg_scores))


	def best_model(self):
		if len(self._models) == 0 or len(self._scores) == 0:
			raise Exception("Not fitted! Call fit before")

		max_index = self._avg_scores.index(max(self._avg_scores))
		return self._models[max_index]

	def best_score(self):
		if len(self._models) == 0 or len(self._scores) == 0:
			raise Exception("Not fitted! Call fit before")

		return max(self._avg_scores)


def get_probabilities(model, X):
	probabilities = model.predict_proba(X)

	return np.array([t[1] if len(t) >= 1 else 0.0 for t in probabilities])

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

def get_best_threshold(model, X, y, metric = f1_score):
	probabilities = get_probabilities(model, X)

	fpr, tpr, threshold = roc_curve(y, probabilities)
	
	def get_score(threshold, probabilities, y_true):
		predicts = [1 if p >= threshold else 0 for p in probabilities]
		return metric(predicts, y_true)

	scores = [get_score(t, probabilities, y) for t in threshold]
	max_index = scores.index(max(scores))

	return threshold[max_index], scores[max_index]




















