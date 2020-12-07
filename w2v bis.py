from Load import *

## Generating w2v model
embedding_dim = 200
tokenizer, w2v_model = generate_w2v_model(X_train_full, X_eval, 200)
vocab_size = len(tokenizer.word_index) + 1

##
embedding_vectors = get_weight_matrix(w2v_model, tokenizer.word_index, embedding_dim)

## Word inputs
w2v_dnn = Sequential()
# Non-traiword2vec.pynable embeddidng layer ?
w2v_dnn.add(Embedding(vocab_size, output_dim=embedding_dim, weights=[embedding_vectors],
                      input_length=20, trainable=False))
# LSTM
w2v_dnn.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.3, dropout=0.3,
                               return_sequences=True)))
w2v_dnn.add(Bidirectional(GRU(units=32, recurrent_dropout=0.1, dropout=0.1)))
w2v_dnn.summary()

## Num imputs
num_input = Input(shape=(9,), name="input_num")
x = Dense(32, activation='relu')(num_input)
num_dnn = Model(inputs=num_input, outputs=x)
num_dnn.summary()

## Combined
combined_inputs = concatenate([num_dnn.output, w2v_dnn.output])
x = Dense(32, activation='relu')(combined_inputs)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='relu')(x)
model = Model(inputs=[w2v_dnn.input, num_dnn.input], outputs=x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_error", optimizer=opt)
model.summary()

##
x_train_full = model_to_tok(X_train_full, tokenizer)
# x_test = model_to_tok(X_test, tokenizer)
x_eval = model_to_tok(X_eval, tokenizer)

##
history = model.fit(x=[x_train_full, Xb_train_full],
                    y=y_train_full,
                    epochs=5,
                    validation_split=0.2,
                    batch_size=128)
##
# model.save('w2v_model.h5')
# ##
# y_pred_eval = model.predict([x_eval, Xb_eval])
# save_it(X_eval, y_pred_eval)



##

