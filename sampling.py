from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import random
import argparse
from tqdm import tqdm


class Sampler:

    def __init__(self, model_checkpoint_path, hidden_state_size, translation_dict_path):
        author = model_checkpoint_path
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(model_checkpoint_path + ".meta")
            saver.restore(self._sess, author)
        self._hidden_state_size = hidden_state_size
        self._h = np.zeros(hidden_state_size)
        self._translation_dict = pickle.load(open(translation_dict_path, "rb"))

    def close(self):
        self._sess.close()

    def reset_hidden(self):
        self._h = np.zeros(self._hidden_state_size)

    def sample_length(self, length, seed, reset_hidden=False):
        out_str = ""
        x = self.__init_variables(seed, reset_hidden)
        for _ in range(length):
            x, outchar = self.__predict(x)
            out_str += outchar
        return out_str

    def sample_till_char(self, seed, character="\n", reset_hidden=False):
        out_str = ""
        outchar = ""
        x = self.__init_variables(seed, reset_hidden)
        while outchar != character:
            x, outchar = self.__predict(x)
            out_str += outchar
        return out_str

    def get_loss(self, x, y, batchsize):
        feed_dict = {"input_preparation/X:0": x,
                     "input_preparation/y:0": y,
                     "parameters/dropout_prob:0": 1.0,
                     "input_preparation/initial_state:0": self._h,
                     "parameters/batchsize:0": batchsize}
        loss, bl = self._sess.run(
            ["Mean_1:0", "batch_loss:0"], feed_dict=feed_dict)
        return loss, bl

    def get_dist(self, seed, reset_hidden):
        x = self.__init_variables(seed, reset_hidden)
        return self.__predict(x, -1)

    def sample_single_char(self, seed, reset_hidden=False, sample_function=0):
        x = self.__init_variables(seed, reset_hidden)
        x, outchar = self.__predict(x, sample_function)
        return outchar

    def __init_variables(self, seed, reset_hidden=False):
        str_int_list = [self._translation_dict["CHAR_TO_INT"][ch]
                        for ch in seed]
        x = np.array([str_int_list])
        if reset_hidden:
            self._h = np.zeros(self._hidden_state_size)
        return x

    def __predict(self, x, sample_function=0):
        ryo, self._h = self._sess.run(["Yo:0", "H:0"], feed_dict={"input_preparation/X:0": x,
                                                                  "parameters/dropout_prob:0": 1.0,
                                                                  "input_preparation/initial_state:0": self._h,
                                                                  "parameters/batchsize:0": 1})
        if sample_function == 0:
            pos = self.sample_by_prop(ryo, len(self._translation_dict["INT_TO_CHAR"]))        
        else:
            return ryo
        outchar = self._translation_dict["INT_TO_CHAR"][pos]
        x = np.array([[pos]])
        return x, outchar

    @staticmethod
    def sample_by_prop(y, vocab_size, count=0):
        if count == 0:
            count = vocab_size
        p = np.squeeze(y)
        p[np.argsort(p)[:-count]] = 0
        p = p / np.sum(p)
        return np.random.choice(vocab_size, 1, p=p)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chkpt_meta_file", dest="chkpt_fn", help="the checkpoints meta file", required=True)
    parser.add_argument("--cell", dest="cell_type", choices=[1, 2], type=int, help="cell type: 1=LSTM, 2=GRU", required=True)
    parser.add_argument("--cell_units", dest="cell_units", type=int, help="number of hidden units in the RNN cells", required=True)
    parser.add_argument("--layers", dest="layers", type=int, help="number of layers", required=True)
    parser.add_argument("--translation_dict", dest="trans_dict", help="path to the saved translation dictionary")
    args =parser.parse_args()

    if args.cell_type == 1:
        hidden_state_size = args.cell_units * (args.layers * 2)
    else:
        hidden_state_size = args.cell_units * args.layers
    
    sampler = Sampler(args.chkpt_fn, hidden_state_size, args.trans_dict)

    for _ in range(20):
        print(sampler.sample_till_char("<"))
