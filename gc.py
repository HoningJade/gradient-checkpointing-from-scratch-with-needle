import sys
sys.path.append('./python')

import needle as ndl
import argparse
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256) 
parser.add_argument('--use_gc', action='store_true') 
parser.add_argument('--lazy', action='store_false') 
args = parser.parse_args()

import needle
needle.autograd.LAZY_MODE = args.lazy

print('lazy mode', 'on' if args.lazy else 'off')

use_gc = args.use_gc
if use_gc:
    print('using gradient checkpointing...')

n_epoch = 3
device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=args.batch_size, device=device, dtype="float32")
model = LanguageModel(20, len(corpus.dictionary), hidden_size=args.batch_size, num_layers=1, seq_model='transformer', seq_len=20, device=device, use_gc=use_gc)

start = time.time()

avg_acc, avg_loss, memo = train_ptb(model, train_data, seq_len=20, n_epochs=n_epoch, device=device, lr=0.003, optimizer=ndl.optim.Adam)

time_used = time.time() - start
# evaluate_ptb(model, train_data, seq_len=20, device=device)

result = {
    'time': time_used / n_epoch,  'batch_size': args.batch_size, 'peak memory': max(memo), 'use_gc': args.use_gc
}

print(result)

import json

if args.use_gc:
    with open(f'result_{args.batch_size}_gc.json', 'w') as f:
        json.dump(result, f)
else:
    with open(f'result_{args.batch_size}.json', 'w') as f:
        json.dump(result, f)