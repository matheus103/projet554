from Load import *

max_features = 120000
maxlen = 200

##
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train_full['text'])
tokenized_train = tokenizer.texts_to_sequences(X_train_full['text'])
x_train_full = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
##
tokenized_test = tokenizer.texts_to_sequences(X_eval['text'])
x_eval = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
##
EMBEDDING_FILE = 'data/glove.txt'

##
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


##
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' '))
                        for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
# change below line if computing normal stats is too slow
embedding_matrix = np.random.normal(emb_mean, emb_std,
                                    (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

batch_size = 128
epochs = 2
embed_size = 200

## Defining Neural Network
gloVe_model = Sequential()
# Non-trainable embeddidng layer
gloVe_model.add(Embedding(nb_words,
                          output_dim=embed_size,
                          weights=[embedding_matrix],
                          input_length=200,
                          trainable=False))
# LSTM
gloVe_model.add(Bidirectional(LSTM(units=128,
                                   recurrent_dropout=0.5,
                                   dropout=0.5)))
gloVe_model.add(Dense(1, activation='sigmoid'))

gloVe_model.summary()

## Num imputs
num_input = Input(shape=(9,), name="input_num")
x = Dense(32, activation='relu')(num_input)
num_dnn = Model(inputs=num_input, outputs=x)
num_dnn.summary()

## Combined
combined_inputs = concatenate([num_dnn.output, gloVe_model.output])
x = Dense(32, activation='relu')(combined_inputs)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='relu')(x)
model = Model(inputs=[gloVe_model.input, num_dnn.input], outputs=x)
opt = Adam(lr=1e-2)
model.compile(loss="mean_absolute_error", optimizer=opt)
model.summary()

##
to_numeric(Xb_train_full)
to_numeric(Xb_eval)
##
history = model.fit(x=[x_train_full, Xb_train_full],
                    y=y_train_full,
                    validation_split=0.2,
                    epochs=2,
                    batch_size=64)
##
model.save('gloVe_model.h5')
##
y_pred_eval = model.predict([x_eval, Xb_eval])
save_it(X_eval, y_pred_eval)

##
with open('data/glove.txt', 'r') as f:
    print('ah')
##

