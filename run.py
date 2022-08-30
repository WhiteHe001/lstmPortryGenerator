import argparse
from dataset.preProcess import preProcess
from Trainer import Trainer

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', type=int, help="批次大小", default=300)
parse.add_argument('--preprocess', action='store_true', help="是否进行预处理")
parse.add_argument('--epochs', type=int, help="训练轮次", default=100)
parse.add_argument('--mode', type=str, choices=['train', 'val'], default='train')
args = parse.parse_args()

if __name__ == "__main__":
    if args.preprocess:
        preProcess()
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
        # trainer.val()
    elif args.mode == 'val':
        trainer.val(True)
