import argparse
import copy, json, os

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.model import BiDAF
from model.data import SQuAD
from model.ema import EMA
import evaluate


def train(args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = BiDAF(args, data.WORD.vocab.vectors).to(device)

    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_exact, max_dev_f1 = -1, -1

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch

        p1, p2 = model(batch)

        optimizer.zero_grad()
        batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
        loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.update(name, param.data)

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_exact, dev_f1 = test(model, ema, args, data)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('exact_match/dev', dev_exact, c)
            writer.add_scalar('f1/dev', dev_f1, c)
            print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
                  f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                max_dev_exact = dev_exact
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

    writer.close()
    print(f'max dev EM: {max_dev_exact:.3f} / max dev F1: {max_dev_f1:.3f}')

    return best_model


def test(model, ema, args, data):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
            p1, p2 = model(batch)
            batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(1)

            for i in range(batch_size):
                id = batch.id[i]
                answer = batch.c_word[0][i][s_idx[i]:e_idx[i]+1]
                answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
                answers[id] = answer

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))

    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    results = evaluate.main(args)
    return loss, results['exact_match'], results['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'dataset_file', f'.data/squad/{args.dev_file}')
    setattr(args, 'prediction_file', f'prediction{args.gpu}.out')
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    print('data loading complete!')

    print('training start!')
    best_model = train(args, data)
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), f'saved_models/BiDAF_{args.model_time}.pt')
    print('training finished!')


if __name__ == '__main__':
    main()
