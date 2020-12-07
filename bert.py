from Load import *
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tokenization

##
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for _text in texts:
        _text = tokenizer.tokenize(_text)

        _text = _text[:max_len - 2]
        input_sequence = ["[CLS]"] + _text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='mean_absolute_error', metrics=['accuracy'])

    return model


##
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=False)

##
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

## !! Attention c'est long ~2-5 min !!
x_train_full = bert_encode(X_train_full.text.values, tokenizer, max_len=160)
x_eval = bert_encode(X_eval.text.values, tokenizer, max_len=160)

##
bert_model = build_model(bert_layer, max_len=160)
bert_model.summary()

## Num imputs
num_input = Input(shape=(9,), name="input_num")
x = Dense(32, activation='relu')(num_input)
x = Dense(32, activation='relu')(x)
num_dnn = Model(inputs=num_input, outputs=x)
num_dnn.summary()

## Combined
combined_inputs = concatenate([num_dnn.output, bert_model.output])
x = Dense(32, activation='relu')(combined_inputs)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='relu')(x)
model = Model(inputs=[bert_model.input, num_dnn.input], outputs=x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
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
model.save('bert_model.h5')
##
y_pred_eval = model.predict([x_eval, Xb_eval])
save_it(X_eval, y_pred_eval)
