from Load import *



##
clf_0_test = MLPClassifier(max_iter=300, early_stopping=True, n_iter_no_change=20, hidden_layer_sizes=(30, 60, 30))
z_pred = test_it_bin(Xb_train, y_train, Xb_test, y_test, clf_0_test)

## ML classifier puis regressor... test
z_pred = clf_0_test.predict(Xb_test)

Xt_train_pos = Xt_train.loc[np.array(X_train['id'])[y_train != 0]]
y_train_pos = y_train.loc[np.array(X_train['id'])[y_train != 0]]
Xt_test_pos = Xt_test.loc[np.array(X_test['id'])[z_pred == 1]]

from sklearn.ensemble import ExtraTreesRegressor
# reg_ml_pos = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=5)
reg_ml_pos = ExtraTreesRegressor(n_estimators=200)
y_pred_pos = test_it(Xt_train_pos, y_train_pos, Xt_test_pos, None, reg_ml_pos, eval_err=False)

y_pred = np.zeros(y_test.shape[0])
y_pred[z_pred == 1] = y_pred_pos
print(mae(y_pred, y_test))

## Save reg_ml_pos
pk.dump(reg_ml_pos, open('var/mlp0/reg_ml_pos.p', 'wb'))
pk.dump(y_pred, open('var/mlp0/y_pred.p', 'wb'))
pk.dump(y_eval_pred, open('var/mlp0/y_eval_pred.p', 'wb'))


## reg_ml_pos load
reg_ml_pos = pk.load(open('var/mlp0/reg_ml_pos.p', 'rb'))
y_pred = pk.load(open('var/mlp0/y_pred.p', 'rb'))


## ML classifier puis regressor... 200 estimators au lieu de 100

z_pred = clf_0_test.predict(Xb_test)

Xt_train_pos = Xt_train.loc[np.array(X_train['id'])[y_train != 0]]
y_train_pos = y_train.loc[np.array(X_train['id'])[y_train != 0]]
Xt_test_pos = Xt_test.loc[np.array(X_test['id'])[z_pred == 1]]

reg_ml_pos_500 = GradientBoostingRegressor(loss='lad', n_estimators=200)
y_pred_pos = test_it(Xt_train_pos, y_train_pos, Xt_test_pos, None, reg_ml_pos_500, eval_err=False)

y_pred_500 = np.zeros(y_test.shape[0])
y_pred[z_pred == 1] = y_pred_pos
print(mae(y_pred, y_test))



## ML classifier puis regressor... eval
from sklearn.ensemble import ExtraTreesRegressor
clf_0_eval = MLPClassifier(max_iter=300, early_stopping=True, n_iter_no_change=20, hidden_layer_sizes=(30, 60, 30))
clf_0_eval.fit(Xb_train_full, y_train_full > 0)
z_pred = clf_0_eval.predict(Xb_eval)

##
plt.plot(clf_0_eval.loss_curve_)
plt.show()
##
Xb_train_full_pos = Xb_train_full.loc[np.array(X_train_full['id'])[y_train_full != 0]]
y_train_full_pos = y_train_full.loc[np.array(X_train_full['id'])[y_train_full != 0]]
Xb_eval_pos = Xb_eval.loc[z_pred == 1]

reg_pos_eval = ExtraTreesRegressor(n_estimators=200)
y_pred_pos = eval_it(Xb_train_full_pos, np.log(y_train_full_pos+0.00001), Xb_eval_pos, X_eval, reg_pos_eval, save=False)

y_pred_pos = np.exp(y_pred_pos)-0.00001
y_pred_pos = np.around(y_pred_pos).astype(int)

print(y_pred_pos)

y_pred = np.zeros(X_eval.shape[0])
y_pred[z_pred == 1] = y_pred_pos
save_it(X_eval, y_pred)



##

