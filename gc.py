import argparse
import json
import sys
import time

sys.path.append("./python")
sys.path.append("./apps")
import needle as ndl
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb


# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--use_gc", action="store_true")
parser.add_argument("--lazy", action="store_false")
args = parser.parse_args()

ndl.autograd.LAZY_MODE = args.lazy
print("lazy mode: ", "on" if args.lazy else "off")

use_gc = args.use_gc
if use_gc:
    print("using gradient checkpointing...")


# Setup Model
## fixed argument values
device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(
    corpus.train, batch_size=args.batch_size, device=device, dtype="float32"
)
n_epoch = 3
embedding_size = 20
output_size = len(corpus.dictionary)
num_layers = 1
seq_len = 20

model = LanguageModel(
    embedding_size=embedding_size,
    output_size=output_size,
    hidden_size=args.batch_size,
    num_layers=num_layers,
    seq_model="transformer",
    seq_len=seq_len,
    device=device,
    use_gc=use_gc,
)

start = time.time()

avg_acc, avg_loss, memo = train_ptb(
    model,
    train_data,
    seq_len=seq_len,
    n_epochs=n_epoch,
    device=device,
    lr=0.003,
    optimizer=ndl.optim.Adam,
)

time_used = time.time() - start
# evaluate_ptb(model, train_data, seq_len=20, device=device)

result = {
    "time": time_used / n_epoch,
    "batch_size": args.batch_size,
    "peak memory": max(memo),
    "use_gc": args.use_gc,
}

print(result)
with open(f"result_{args.batch_size}{'_gc' if use_gc else ''}.json", "w") as f:
    json.dump(result, f)
