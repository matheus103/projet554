from Load import *

##
clf_0_test = MLPClassifier(max_iter=300)
z_pred = test_it_bin(Xb_train, y_train, Xb_test, y_test, clf_0_test)

## ML classifier puis regressor... test
z_pred = clf_0_test.predict(Xb_test)

Xt_train_pos = Xt_train.loc[np.array(X_train['id'])[y_train != 0]]
y_train_pos = y_train.loc[np.array(X_train['id'])[y_train != 0]]
Xt_test_pos = Xt_test.loc[np.array(X_test['id'])[z_pred == 1]]

reg_ml_pos = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=5)
y_pred_pos = test_it(Xt_train_pos, y_train_pos, Xt_test_pos, None, reg_ml_pos, eval_err=False)

y_pred = np.zeros(y_test.shape[0])
y_pred[z_pred == 1] = y_pred_pos
print(mae(y_pred, y_test))

## Save reg_ml_pos
pk.dump(reg_ml_pos, open('var/reg/classReg/reg_ml_pos.p', 'wb'))
pk.dump(y_pred, open('var/reg/classReg/y_pred.p', 'wb'))
# pk.dump(y_eval_pred, open('var/reg/gbr/y_eval_pred.p', 'wb'))

## reg_ml_pos load
reg_ml_pos = pk.load(open('var/reg/classReg/reg_ml_pos.p', 'rb'))
y_pred = pk.load(open('var/reg/classReg/y_pred.p', 'rb'))

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
clf_0_eval = MLPClassifier(max_iter=300, early_stopping=True, hidden_layer_sizes=(30, 30, 30))
clf_0_eval.fit(Xb_train_full, y_train_full > 0)
z_pred = clf_0_eval.predict(Xb_eval)

##
plt.plot(clf_0_eval.loss_curve_)
plt.show()
##
Xb_train_full_pos = Xb_train_full.loc[np.array(X_train_full['id'])[y_train_full != 0]]
y_train_full_pos = y_train_full.loc[np.array(X_train_full['id'])[y_train_full != 0]]
Xb_eval_pos = Xb_eval.loc[z_pred == 1]

reg_pos_eval = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth=5)
y_pred_pos = eval_it(Xb_train_full_pos, y_train_full_pos, Xb_eval_pos, X_eval, reg_pos_eval, save=False)

y_pred = np.zeros(X_eval.shape[0])
y_pred[z_pred == 1] = y_pred_pos
save_it(X_eval, y_pred)

##


## Analyse de la répartition de l'erreur

y = np.zeros((2, 199734))
y[0] = y_test
y[1] = y_pred
y = y.T


# for k in range(100):
#     print(y[k])

##
def maes(y, end=0):
    n = y.shape[0]
    a = 0
    b = int(y[:, 0].max())
    if end > 0:
        b = end
    dx = b // 200
    r = np.arange(a, b, dx)
    for s in range(a, b, dx):
        r[s // dx] = np.sum(np.abs(y[:, 0][y[:, 0] < s] - y[:, 1][y[:, 0] < s])) / n
    r = 100 * r / mae(y[:, 0], y[:, 1])
    plt.plot(np.arange(a, b, dx), r)
    # plt.xlim(0, 10)
    plt.show()
    return np.floor(r)


def maes_freq(y, end=0):
    n = y.shape[0]
    a = 0
    b = int(y[:, 0].max())
    if end > 0:
        b = end
    dx = b // 200
    r = np.arange(a, b, dx)
    for s in range(a, b, dx):
        r[s // dx] = np.sum(
            np.abs(y[:, 0][(y[:, 0] > s) * (y[:, 0] < s + dx)] - y[:, 1][(y[:, 0] > s) * (y[:, 0] < s + dx)])) / n
    r = 100 * r / mae(y[:, 0], y[:, 1])
    plt.plot(np.arange(a, b, dx), r)
    # plt.xlim(0, 10)
    plt.show()
    return np.floor(100 * r) / 100


def maes2(y, end=0):
    n = y.shape[0]
    a = 0
    b = int(y[:, 0].max())
    if end > 0:
        b = end
    dx = b // 200
    r = np.arange(a, b, dx)
    for s in range(a, b, dx):
        r[s // dx] = np.sum(np.abs(y[:, 0][y[:, 1] < s] - y[:, 1][y[:, 1] < s])) / n
    r = 100 * r / mae(y[:, 0], y[:, 1])
    plt.plot(np.arange(a, b, dx), r)
    # plt.xlim(0, 10)
    plt.show()
    return np.floor(r)


def maes_freq2(y, end=0):
    n = y.shape[0]
    a = 0
    b = int(y[:, 0].max())
    if end > 0:
        b = end
    dx = b // 200
    r = np.arange(a, b, dx)
    for s in range(a, b, dx):
        r[s // dx] = np.sum(
            np.abs(y[:, 0][(y[:, 1] > s) * (y[:, 1] < s + dx)] - y[:, 1][(y[:, 1] > s) * (y[:, 1] < s + dx)])) / n
    r = 100 * r / mae(y[:, 0], y[:, 1])
    plt.plot(np.arange(a, b, dx), r)
    # plt.xlim(0, 10)
    plt.show()
    return np.floor(100 * r) / 100


##
maes_freq2(y, 10000)

##
maes_freq(y)

##
maes(y, 100000)
##
mae(y[:, 0][y[:, 0] > 10000], y[:, 1][y[:, 0] > 10000])

##
plt.hist(y_pred, bins=30)
plt.yscale('log')
plt.legend()
plt.show()

## Save the classificator and regressor for eval
pk.dump(clf_0_eval, open('var/reg/mlp0/clf_0_eval.p', 'wb'))
pk.dump(reg, open('var/reg/classReg/reg_pos_eval.p', 'wb'))
# pk.dump(y_eval_pred, open('var/reg/gbr/y_eval_pred.p', 'wb'))

## reg_ml_pos load
# clf_0_eval = pk.load(open('var/reg/mlp0/clf_0_eval.p', 'rb'))
# reg_pos_eval = pk.load(open('var/reg/classReg/reg_pos_eval.p', 'rb'))

## Complexité GBR :
res = np.zeros(6)
for n_es in range(1, 6):
    t0 = time.time()
    regreg = GradientBoostingRegressor(loss='lad', n_estimators=n_es, max_depth=5)
    regreg.fit(Xt_train, y_train)
    res[n_es] = time.time() - t0
    print(time.time() - t0)
plt.plot(np.arange(6), res)
plt.show()
##
t0 = time.time()
regreg = GradientBoostingRegressor(loss='lad', n_estimators=1, max_depth=5, criterion='mae')
regreg.fit(Xt_train, y_train)
print(time.time() - t0)
##
