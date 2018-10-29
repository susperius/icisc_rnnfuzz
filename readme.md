# Recurrent neural networks for fuzz testing web browsers #

This repository contains the code used for the paper "Recurrent neural networks for fuzz testing web browsers", which was
accepted as conference for the 21st annual International Conference on Information Security and Cryptology (ICISC). The 
paper was published in Springer's "Lecture Notes in Computer Science".

## Usage ##
###  Training ###

```bash
usage: stacked_rnn.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                      [--learning_rate LEARNING_RATE] [--layers LAYERS]
                      [--internal_size INTERNAL_SIZE] [--seq_len SEQ_LEN]
                      [--training_set TRAINING_SET] [--load_size LOAD_SIZE]
                      [--out_folder OUT_FOLDER] [--split SPLIT]
                      [--cells {1,2}]

This script trains a stacked RNN model

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        size of the batch
  --epochs EPOCHS       number of training epochs
  --learning_rate LEARNING_RATE
                        learning rate
  --layers LAYERS       number of RNN layers
  --internal_size INTERNAL_SIZE
                        number of nodes inside a RNN cell
  --seq_len SEQ_LEN     sequence length used for training
  --training_set TRAINING_SET
                        path to the used training set
  --load_size LOAD_SIZE
                        defines how much data is loaded from the file in bytes
  --out_folder OUT_FOLDER
                        defines an output folder
  --split SPLIT         defines which split to use
  --cells {1,2}         sets the cell type to use: 1=LSTM, 2=GRU
```

This script trains the specified model. It creates tensorboard log events for validation loss, batch loss and accuracy. 
A checkpoint is generated for each epoch of training and each time a new lowest validation loss is achieved. In addition,
a small sample is generated after each epoch of training.

 ### Sampling ###

```bash
usage: sampling.py [-h] --chkpt_meta_file CHKPT_FN --cell {1,2} --cell_units
                   CELL_UNITS --layers LAYERS [--translation_dict TRANS_DICT]

required arguments:
  --chkpt_meta_file CHKPT_FN
                        the checkpoints meta file
  --cell {1,2}          cell type: 1=LSTM, 2=GRU
  --cell_units CELL_UNITS
                        number of hidden units in the RNN cells
  --layers LAYERS       number of layers
  --translation_dict TRANS_DICT
                        path to the saved translation dictionary
optional arguments:
  -h, --help            show this help message and exit
```
This script demonstrates how a trained model can be used to generate new HTML-tags. It uses the provided model to sample
20 from it and prints the results.