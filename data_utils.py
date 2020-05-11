import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.preprocessing import text, sequence
import io
from config import enc_vocab_size , max_length_inp ,dec_vocab_size



def create_dataset(path, num_examples=None):
    all_data_list = io.open(path, encoding='utf-8').read().strip().split("\n")
    inputs, targets = [], []
    if num_examples :
        all_data_list = all_data_list[:num_examples]
    for ix, line in enumerate(all_data_list):
        if ix % 2 == 0:
            inputs.append(line)
        else:
            targets.append(line)
    return  inputs , targets


def max_length(tensor):
    return max(len(t) for t in tensor)


def read_data(path, num_examples):
    input_lang, target_lang = create_dataset(path, num_examples)

    input_tensor ,input_index_word , input_word_index , input_vocab ,_= tokenize(input_lang ,enc_vocab_size )
    target_tensor ,target_index_word , target_word_index , target_vocab ,_ = tokenize(target_lang ,dec_vocab_size )

    return input_tensor ,input_index_word , input_word_index , input_vocab  , target_tensor ,target_index_word , target_word_index , target_vocab


def tokenize(lang , vocab_size =None  ):
    pad_value = 0
    startvalue = 1
    endvalue = 2
    unkvalue = 3
    words ={}

    import  jieba
    tensors  =[]
    for  line in lang :
        l = jieba.lcut( line)
        tensors.append( l )
        for w in l:
            words.setdefault( w , 0 )
            words[w] +=1
    if vocab_size:
        vocab_pari= sorted( words.items() , key= lambda  x :-x[1])[: vocab_size - 4 ]
    else:
        vocab_pari = sorted(words.items, key=lambda x: -x[1])
    vocab = [ v for  v  ,_ in vocab_pari ]


    word_index =  { k:v+4 for k ,v in zip(  vocab , range(len(vocab)))}
    word_index.update( { "pad" : pad_value  , "start" :startvalue ,'end' : endvalue ,"unk" : unkvalue}  )
    index_word = { v:k for k,v in word_index.items()}

    vocab.extend( [ "pad", "start"  , 'end' ,"unk" ])

    X = []
    for line in tensors :
        ids =[]
        for w in line :
            ids.append( word_index.get( w, word_index['unk']) )
        X.append( ids)
    X = sequence.pad_sequences( X, padding= 'post' , value= pad_value , maxlen= max_length_inp )


    return np.array(X) ,index_word , word_index , vocab , len(vocab )




if __name__ == '__main__':
    inputs , targets = create_dataset('new_corpus.txt' , num_examples=101)
    # print ( inputs , targets )
    for  i, t in zip(  inputs, targets ) :
        print ( i , '============>'  , t )

    input_tensor ,input_index_word , input_word_index , input_vocab  , target_tensor ,target_index_word , target_word_index , target_vocab = read_data('new_corpus.txt'  , num_examples= 100 )

    print ( input_tensor.shape)
    print ( target_tensor.shape )

    print( input_tensor[:30])

    for  input_line  in input_tensor[:30]:
        print ( [ input_index_word[i] for i in input_line ])
