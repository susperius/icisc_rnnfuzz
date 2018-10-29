"""
This file is the training script for the stackedRNN approach. It allows the use
of GRU cell or LSTM cells. It was created by following Martin Goerner's tutorial 
and example code available at https://github.com/martin-gorner/tensorflow-rnn-shakespeare
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle
import os
import input_preperation as ip
import math
import time
import argparse
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tqdm import tqdm
from sampling import Sampler


TRAINING_SET = "training_set/training_set.html"


DEFAULT_TRAINING_SET = TRAINING_SET
DEFAULT_SIZE_TO_LOAD = 30 * 10 ** 6

SEQ_LEN = 150
BATCH_SIZE = 512
EPOCHS = 50
ALPHA = 0.001
LAYERS = 1
INTERNALSIZE = 256
LSTM = True


def load_training_set(path_to_training_set, size_to_load=0, save=True):
    training_set, translation = ip.load_input_on_char_basis(path_to_training_set, size_to_load, True)
    pickle.dump(translation, open("trans_dict.pickle", "wb"))
    if save:
        np.save("training_set_np", training_set)
    return training_set, translation


def load_from_file():
    translation = pickle.load(open("trans_dict.pickle", "rb"))
    training_set = np.load("training_set_np.npy")
    return training_set, translation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains a stacked RNN model")
    parser.add_argument("--batch_size", dest="batch_size", help="size of the batch", default=BATCH_SIZE, type=int)
    parser.add_argument("--epochs", dest="epochs", help="number of training epochs", default=EPOCHS, type=int)
    parser.add_argument("--learning_rate", dest="learning_rate", help="learning rate", default=ALPHA, type=float)
    parser.add_argument("--layers", dest="layers", help="number of RNN layers", default=LAYERS, type=int)
    parser.add_argument("--internal_size", dest="internal_size", help="number of nodes inside a RNN cell",
                        default=INTERNALSIZE, type=int)
    parser.add_argument("--seq_len", dest="seq_len", help="sequence length used for training", default=SEQ_LEN,
                        type=int)
    parser.add_argument("--training_set", dest="training_set", help="path to the used training set",
                        default=DEFAULT_TRAINING_SET)
    parser.add_argument("--load_size", dest="load_size", help="defines how much data is loaded from the file in bytes",
                        type=int, default=DEFAULT_SIZE_TO_LOAD)
    parser.add_argument("--out_folder", dest="out_folder", help="defines an output folder", default="./")
    parser.add_argument("--split", dest="split", help="defines which split to use", type=int, default=0)
    parser.add_argument("--cells", dest="cell_type", choices=[1,2], type=int, help="sets the cell type to use: 1=LSTM, 2=GRU", default=1)
    args = parser.parse_args()
    print("[+] Init training set and translation dict ...")
    if os.path.isfile("trans_dict.pickle") and os.path.isfile("training_set_np.npy"):
        training_set, translation = load_from_file()
    else:
        training_set, translation = load_training_set(args.training_set, args.load_size)
    print("[*] Total input length: " + str(training_set.shape[0]))
    print("[*] Vocabulary size: " + str(len(translation["CHAR_TO_INT"])))
    if not os.path.isfile("vali_set.npy"):
        vali_set, _ = ip.load_input_on_char_basis("training_set/vali_set.html", length=10 ** 6,
                                                  int_based=True, trans_dict=translation)
        np.save("vali_set", vali_set)
    else:
        vali_set = np.load("vali_set.npy")
    if args.split != 0:
        training_set = np.append(training_set, vali_set)
        if args.split == 1:
            vali_set = training_set[21 * 10 ** 6:22 * 10 ** 6]
            training_set = np.append(training_set[:21 * 10 ** 6], training_set[22 * 10 ** 6:])
        elif args.split == 2:
            vali_set = training_set[17 * 10 ** 6:18 * 10 ** 6]
            training_set = np.append(training_set[:17*10**6], training_set[18 * 10 ** 6:])
        elif args.split == 3:
            vali_set = training_set[5 * 10 ** 6:6 * 10 ** 6]
            training_set = np.append(training_set[:5 * 10 ** 6], training_set[6 * 10 ** 6:])
        elif args.split == 4:
            vali_set = training_set[23 * 10 ** 6:24 * 10 ** 6]
            training_set = np.append(training_set[:23 * 10 ** 6], training_set[24 * 10 ** 6:])
        elif args.split == 5:
            vali_set = training_set[28 * 10 ** 6:29 * 10 ** 6]
            training_set = np.append(training_set[:28 * 10 ** 6], training_set[29 * 10 ** 6:])

    if not os.path.exists(os.path.join(args.out_folder, "samples")):
        os.makedirs(os.path.join(args.out_folder, "samples"))


    VOCAB_SIZE = len(translation["CHAR_TO_INT"].keys())
    with tf.name_scope("parameters"):
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
        batchsize = tf.placeholder(tf.int32, name="batchsize")
        alpha = tf.placeholder(tf.float32, name="alpha")

    with tf.name_scope("input_preparation"):
        X = tf.placeholder(tf.uint8, [None, None], name="X")
        Xo = tf.one_hot(X, VOCAB_SIZE, 1.0, 0.0)

        Y_ = tf.placeholder(tf.uint8, [None, None], name="y")
        Yo_ = tf.one_hot(Y_, VOCAB_SIZE, 1.0, 0.0)
        if args.cell_type == 1:
            initial_state = tf.placeholder(tf.float32, [None, args.internal_size * (args.layers * 2)], name="initial_state")
        else:
            initial_state = tf.placeholder(tf.float32, [None, args.internal_size * (args.layers)], name="initial_state")

    if args.cell_type == 1:
        net = [rnn.BasicLSTMCell(args.internal_size, state_is_tuple=False) for _ in range(args.layers)]
    else:
        net = [rnn.GRUCell(args.internal_size) for _ in range(args.layers)]
    net = [rnn.DropoutWrapper(cell, input_keep_prob=dropout_prob) for cell in net]

    multi_rnn = rnn.MultiRNNCell(net, state_is_tuple=False)
    drop_multi_rnn = rnn.DropoutWrapper(multi_rnn, output_keep_prob=dropout_prob)

    Yr, H = tf.nn.dynamic_rnn(drop_multi_rnn, Xo, initial_state=initial_state, dtype=tf.float32)

    H = tf.identity(H, name="H")

    Yflat = tf.reshape(Yr, [-1, args.internal_size])
    Ylogits = layers.linear(Yflat, VOCAB_SIZE)

    Yflat_ = tf.reshape(Yo_, [-1, VOCAB_SIZE])


    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,
                                                   labels=Yflat_)  
    loss = tf.reshape(loss, [batchsize, -1])

    Yo = tf.nn.softmax(Ylogits, name="Yo")
    Y = tf.argmax(Yo, 1)
    Y = tf.reshape(Y, [batchsize, -1], name="Y")

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    if args.cell_type == 1:
        in_state = np.zeros([args.batch_size, args.internal_size * (args.layers * 2)])
    else:
        in_state = np.zeros([args.batch_size, args.internal_size * (args.layers)])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    graph_writer = tf.summary.FileWriter(args.out_folder + "logs/", graph=sess.graph)
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter(args.out_folder + "logs/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter(args.out_folder + "logs/" + timestamp + "-validation")
    saver = tf.train.Saver(max_to_keep=1000)

    learn_rate = args.learning_rate
    step = 0
    act_epoch = -1
    steps_per_epoch = (training_set.shape[0] - 1) // (args.batch_size * args.seq_len)
    actual_vl = 0
    min_vl = 10000

    BATCH_LOSS_FREQ = 30
    VALI_LOSS_FREQ = 30
    SAVE_FREQ = steps_per_epoch
    DISPL_FREQ = 10
    SAMPLE_FREQ = steps_per_epoch
    _50_BATCHES = DISPL_FREQ * args.batch_size * args.seq_len
    LAST_DIVISION = 0
    VALI_SEQLEN = 150  
    vl_bsize = 512  
    vl_set_sequencer = ip.rnn_minibatch_sequencer(vali_set, vl_bsize, VALI_SEQLEN, 1000)
    for x, y_, epoch in ip.rnn_minibatch_sequencer(training_set, args.batch_size, args.seq_len, args.epochs):
        if act_epoch != epoch:
            act_epoch = epoch
            prog_bar = tqdm(total=steps_per_epoch, desc="Epoch " + str(act_epoch + 1) + " of " + str(args.epochs))
        feed_dict = {X: x, Y_: y_, initial_state: in_state, learning_rate: learn_rate,
                     dropout_prob: 0.5, batchsize: args.batch_size}
        _, y, ostate, bl = sess.run([train_step, Y, H, batchloss], feed_dict=feed_dict)
        if step % BATCH_LOSS_FREQ == 0:
            feed_dict = {X: x, Y_: y_, initial_state: in_state, dropout_prob: 1.0,
                         batchsize: args.batch_size}  # no dropout for validation
            y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
            summary_writer.add_summary(smm, step)
        if ((step % VALI_LOSS_FREQ == 0) or (step % SAVE_FREQ == 0)) and len(vali_set) > 0:  # vl also for checkpoints
            vali_x, vali_y, _ = next(vl_set_sequencer)
            if args.cell_type == 1:
                vali_nullstate = np.zeros([vl_bsize, args.internal_size * (args.layers * 2)])
            else:
                vali_nullstate = np.zeros([vl_bsize, args.internal_size * (args.layers)])
            feed_dict = {X: vali_x, Y_: vali_y, initial_state: vali_nullstate, dropout_prob: 1.0,
                         # no dropout for validation
                         batchsize: vl_bsize}
            actual_vl, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
            if actual_vl < min_vl:
                min_vl = actual_vl
                saver.save(sess, args.out_folder + "checkpoints/rnn_train_lowest_vl")
            # save validation data for Tensorboard
            validation_writer.add_summary(smm, step)
        if step % SAVE_FREQ == 0:
            saver.save(sess, args.out_folder + "checkpoints/ep_" + str(act_epoch + 1) + "_rnn_train_" + timestamp + "_vl_" +
                       str(actual_vl), global_step=step)
        if step % steps_per_epoch == 0:
            ry = np.array([[0]])
            if args.cell_type == 1:
                rh = np.zeros([1, args.internal_size * (args.layers * 2)])
            else:
                rh = np.zeros([1, args.internal_size * (args.layers)])
            out_str = ""
            for k in range(1000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, dropout_prob: 1.0, initial_state: rh, batchsize: 1})
                char_int_list = []
                for i in range(1):
                    char_int = Sampler.sample_by_prop(ryo[i], VOCAB_SIZE, 0)
                    out_str += translation["INT_TO_CHAR"][char_int]
                    char_int_list.append(char_int)
                ry = np.array([char_int_list])
            with open(args.out_folder + "samples/sample_" + timestamp + "_epoch_" + str(epoch + 1) + ".txt", "wb") as fd:
                fd.write("======================================== SAMPLE ========================================\n")
                fd.write(out_str)
                fd.write("\n====================================== END SAMPLE ======================================\n")
        if (epoch + 1) % 10 == 0 and LAST_DIVISION != (epoch + 1):
            learn_rate /= 2  # from the seq2seq paper but a wee bit later epoch wise
            LAST_DIVISION = (epoch + 1)
            # learn_rate = learn_rate - learn_rate_decay if learn_rate - learn_rate_decay > 0 else 0.001
        in_state = ostate
        step += 1
        prog_bar.update(1)
        prog_bar.set_postfix({"tloss": bl, "lr": learn_rate, "min-vl": min_vl})
        ###### END TRAINING FOR LOOP ##########
    if len(vali_set) > 0:
        # VALI_SEQLEN = 1 * 150  # Sequence length for validation. State will be wrong at the start of each sequence.
        # vl_bsize = 512 # len(vali_set) // VALI_SEQLEN
        vali_x, vali_y, _ = next(vl_set_sequencer)  # all data in 1 batch
        if args.cell_type == 1:
            vali_nullstate = np.zeros([vl_bsize, args.internal_size * (args.layers * 2)])
        else:
            vali_nullstate = np.zeros([vl_bsize, args.internal_size * (args.layers)])
        feed_dict = {X: vali_x, Y_: vali_y, initial_state: vali_nullstate, dropout_prob: 1.0,
                     # no dropout for validation
                     batchsize: vl_bsize}
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        validation_writer.add_summary(smm, step)
        with open(args.out_folder + "checkpoints/last_vl_loss.txt", "wb") as fd:
            fd.write("Last Validation loss: " + str(ls))
    saver.save(sess, args.out_folder + "checkpoints/rnn_train_" + timestamp, global_step=step)
    print("===========================================================================================================")
    print("===========================================================================================================")
    print("===========================================================================================================")
    print("===========================================================================================================")
