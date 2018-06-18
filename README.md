# BiDAF-pytorch
Re-implementation of [BiDAF](https://arxiv.org/abs/1611.01603)(Bidirectional Attention Flow for Machine Comprehension, Minjoon Seo et al., ICLR 2017) on PyTorch.

## Results

Dataset: [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)

| Model(Single) | EM(%)(dev) | F1(%)(dev) |
|--------------|:----------:|:----------:|
| **Re-implementation** | **64.8** | **75.7** | 
| Baseline(paper) | 67.7 | 77.3 |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- GPU: Nvidia Titan Xp
- Language: Python 3.6.2.
- Pytorch: **0.4.0**

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    torch==0.4.0
    nltk==3.2.4
    tensorboardX==0.8
    torchtext==0.2.3

## Execution

> python run.py --help

	usage: run.py [-h] [--char-dim CHAR_DIM]
              [--char-channel-width CHAR_CHANNEL_WIDTH]
              [--char-channel-size CHAR_CHANNEL_SIZE]
              [--dev-batch-size DEV_BATCH_SIZE] [--dev-file DEV_FILE]
              [--dropout DROPOUT] [--epoch EPOCH] [--gpu GPU]
              [--hidden-size HIDDEN_SIZE] [--learning-rate LEARNING_RATE]
              [--print-freq PRINT_FREQ] [--train-batch-size TRAIN_BATCH_SIZE]
              [--train-file TRAIN_FILE] [--word-dim WORD_DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --char-dim CHAR_DIM
      --char-channel-width CHAR_CHANNEL_WIDTH
      --char-channel-size CHAR_CHANNEL_SIZE
      --dev-batch-size DEV_BATCH_SIZE
      --dev-file DEV_FILE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --hidden-size HIDDEN_SIZE
      --learning-rate LEARNING_RATE
      --print-freq PRINT_FREQ
      --train-batch-size TRAIN_BATCH_SIZE
      --train-file TRAIN_FILE
      --word-dim WORD_DIM

