# -*- coding:utf-8 -*-
import os
import sys
import time
import tensorflow as tf
import seq2seqModel
import jieba
import pickle
import warnings

warnings.filterwarnings('ignore')

max_length_inp, max_length_tar = 20, 20
vocab_inp_size = 20000
vocab_tar_size = 20000
embedding_dim = 128
units = 256
max_train_data_size = 50000
batch_size = 128

from data_utils import read_data


def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w


def max_length(tensor):
    return max(len(t) for t in tensor)


def train():
    steps_per_epoch = len(input_tensor) // batch_size
    print(steps_per_epoch)
    enc_hidden = seq2seqModel.encoder.initialize_hidden_state()
    checkpoint_dir = 'models'
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    checkpoint_dir = 'models'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()

    while True:
        start_time_epoch = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = seq2seqModel.train_step(inp, targ, target_word_index, enc_hidden)
            total_loss += batch_loss
            print(batch_loss.numpy())

        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps

        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      step_loss.numpy()))
        seq2seqModel.checkpoint.save(file_prefix=checkpoint_prefix)

        sys.stdout.flush()


def predict(sentence):
    checkpoint_dir = 'models'
    seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    sentence = preprocess_sentence(sentence)
    inputs = [input_word_index.get(i, input_word_index['unk']) for i in jieba.lcut(sentence)]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = seq2seqModel.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_word_index['start']], 0)

    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = seq2seqModel.decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        if target_index_word[predicted_id] == 'end':
            break
        result += target_index_word[predicted_id]

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


if __name__ == '__main__':
    # train()
    traing = False
    if traing:
        input_tensor, input_index_word, input_word_index, input_vocab, target_tensor, target_index_word, target_word_index, target_vocab = read_data(
            'new_corpus.txt', num_examples=None)

        meta = {
            'input_tensor': input_tensor,
            'input_index_word': input_index_word,
            'input_word_index': input_word_index,
            'input_vocab': input_vocab,
            'target_tensor': target_tensor,
            'target_index_word': target_index_word,
            'target_word_index': target_word_index,
            'target_vocab': target_vocab

        }

        with open('mypickle.pickle', 'wb') as f:
            pickle.dump(meta, f)
    else:
        with open('mypickle.pickle', 'rb') as f:
            meta = pickle.load(f)
        input_tensor = meta['input_tensor']
        input_index_word = meta['input_index_word']
        input_word_index = meta['input_word_index']
        input_vocab = meta['input_vocab']

        target_tensor = meta['target_tensor']
        target_index_word = meta['target_index_word']
        target_word_index = meta['target_word_index']
        target_vocab = meta['target_vocab']

        print("开始对话：")
        while True:
            print(">", end='')
            input_str = input()
            ans_str = predict(input_str.strip())
            print("<", ans_str)

  