from ChatPotte import *

# X_train, y_train, X_test, y_test, X_train_full, y_train_full, X_eval = \
#     load_it('both')
# ##
# Xb_train, Xb_test, Xb_train_full, Xb_eval = \
#     boost_it([X_train, X_test, X_train_full, X_eval], word_dim=0, tok_dim=0)
#
# print(Xb_train.shape, Xb_test.shape, Xb_train_full.shape, Xb_eval.shape)
# Xt_train, Xt_test, Xt_train_full, Xt_eval = \
#     boost_it([X_train, X_test, X_train_full, X_eval], word_dim=200, tok_dim=10)
#
# X = [X_train, y_train, X_test, y_test, X_train_full, y_train_full, X_eval]
# Xb = [Xb_train, Xb_test, Xb_train_full, Xb_eval]
# Xt = [Xt_train, Xt_test, Xt_train_full, Xt_eval]
#
# ## Saving tables
# pk.dump(X, open('var/tweets.p', 'wb'))
# pk.dump(Xb, open('var/tweets_b.p', 'wb'))
# pk.dump(Xt, open('var/tweets_t.p', 'wb'))

## Loading tables
[X_train, y_train, X_test, y_test, X_train_full, y_train_full, X_eval] = \
    pk.load(open('var/tweets.p', 'rb'))

[Xb_train, Xb_test, Xb_train_full, Xb_eval] = \
    pk.load(open('var/tweets_b.p', 'rb'))
[Xt_train, Xt_test, Xt_train_full, Xt_eval] = \
    pk.load(open('var/tweets_t.p', 'rb'))

clf_0_test = pk.load(open('var/reg/mlp0/clf_0_test.p', 'rb'))

##

