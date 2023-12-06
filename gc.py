import sys
sys.path.append('./python')

import needle as ndl
import argparse
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

parser = argparse.ArgumentParser()
parser.add_argument('--use_gc', action='store_true') 
args = parser.parse_args()

use_gc = args.use_gc
if use_gc:
    print('using gradient checkpointing...')

device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=256, device=device, dtype="float32")
model = LanguageModel(20, len(corpus.dictionary), hidden_size=32, num_layers=1, seq_model='transformer', seq_len=20, device=device, use_gc=use_gc)
train_ptb(model, train_data, seq_len=20, n_epochs=5, device=device, lr=0.003, optimizer=ndl.optim.Adam)
evaluate_ptb(model, train_data, seq_len=20, device=device)