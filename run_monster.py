import utils
import pandas as pd
import numpy as np
import monster
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 23423

params_rf = {
	'n_estimators' : [100, 120],
	'min_samples_leaf' : [4, 5, 6],
	'min_samples_split' : [13, 15, 19],
	'n_jobs' : [-1],
	'random_seed' : [RANDOM_SEED]
}

params_xgb = {
	'base_score' : [0.3],
	'max_depth' : [3, 10, 50],
	'n_estimators' : [100, 200],
	'seed' : [RANDOM_SEED],
	'nthread' : [2]
}

params_ada = {
	'n_estimators' : [50, 70, 100],
	'learning_rate' : [1.0, 0.9, 1.1],
	'random_state' : [RANDOM_SEED]
}

params_mlp = {
	'hidden_layer_sizes' : [(100,), (120,), (100,20)],
	'random_state' : [RANDOM_SEED],
	'early_stopping' : [True]
}


train = pd.read_csv('data/train_all_features.csv')
X = train.drop(['CUST_COD', 'DEFAULT_PAYMENT_JAN'])
y = train['DEFAULT_PAYMENT_JAN']


models = [RandomForestClassifier, XGBClassifier, AdaBoostClassifier, MLPClassifier]
params = [params_rf, params_xgb, params_ada, params_mlp]

mst = monster.Monster(models, params, random_state = RANDOM_SEED)
mst.fit(X, y)

f = open('models_out.pydat', 'wb')
pickle.dump(mst, f)
f.close()
