import argparse
from model.bert import BertConfig
import torch
from reader.cls import MnliDataProcessor
import numpy as np
import random
from model.classifier import Classifier
from utils import load_pickle
import time

from csv_logger import Logger
Logger('./logs')


parser = argparse.ArgumentParser()
# Model
parser.add_argument('--bert_config_path', type=str, default=None,
                    help='Path to the json file for bert model config.')
parser.add_argument('--init_pre_training_params', type=str, default=None,
                    help='Init pre-training params.')
# Training
parser.add_argument('--epoch', type=int, default=3,
                    help='Number of epochs for fine-tuning.')
parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='Learning rate used to train.')
# Logging:
parser.add_argument('--skip_steps', type=int, default=10,
                    help='The steps interval to print loss.')
# parser.add_argument('--verbose', type=bool, default=False,
#                     help='Whether to output verbose log.')

# Data:
parser.add_argument('--data_dir', type=str, default=None,
                    help='Path to training data.')
parser.add_argument('--vocab_path', type=str, default=None,
                    help='Vocabulary path.')
parser.add_argument('--max_seq_len', type=int, default=128,
                    help='Number of words of the longest sequence.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Total examples\' number in batch for training.')
parser.add_argument('--do_lower_case', type=bool, default=True,
                    help='Whether to lower case the input text. '
                         'Should be True for uncased models and False for cased models.')
parser.add_argument('--random_seed', type=int, default=None,
                    help='Random seed.')

# Run type:
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='If set, use GPU for training.')
args = parser.parse_args()


def main(args):
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    processor = MnliDataProcessor(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        random_seed=args.random_seed
    )

    num_labes = len(processor.get_labels())

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='train',
        epoch=args.epoch,
        shuffle=True
    )

    num_train_examples = processor.get_num_examples(phase='train')

    max_train_steps = args.epoch * num_train_examples // args.batch_size

    classifier = Classifier(bert_config, num_labes).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)

    if args.init_pre_training_params:
        pre_training_params = load_pickle(args.init_pre_training_params)
        classifier.bert.load_state_dict(pre_training_params)

        # # Temporal
        # cls_ckp = load_pickle('/home/cvds_lab/maxim/transformer_investigation/notebooks/ckp/classifier_ckp.pkl')
        # classifier.cls_out.weight.data = torch.tensor(cls_ckp['cls_out_w'], dtype=torch.float32).t().to(device)
        # classifier.cls_out.bias.data = torch.tensor(cls_ckp['cls_out_b'], dtype=torch.float32).to(device)
        # # Temporal

    Logger().add_log('train.csv', ['epoch', 'step', 'loss', 'accuracy',
                                   'cls_w_mean', 'cls_w_std', 'cls_w_min', 'cls_w_max',
                                   'cls_b_mean', 'cls_b_std', 'cls_b_min', 'cls_b_max'])
    steps = 0
    total_loss, total_acc = [], []
    time_begin = time.time()
    for batch in train_data_generator():
        steps += 1

        src_ids = torch.tensor(batch[0], dtype=torch.long).to(device)
        position_ids = torch.tensor(batch[1], dtype=torch.long).to(device)
        sentence_ids = torch.tensor(batch[2], dtype=torch.long).to(device)
        input_mask = torch.tensor(batch[3], dtype=torch.float32).to(device)
        labels = torch.tensor(batch[4], dtype=torch.long).to(device)

        optimizer.zero_grad()
        loss, _, accuracy = classifier(src_ids, position_ids, sentence_ids, input_mask, labels)
        loss.backward()
        optimizer.step()

        current_example, current_epoch = processor.get_train_progress()
        Logger()['train.csv']['epoch'].append(current_epoch)
        Logger()['train.csv']['step'].append(steps)
        Logger()['train.csv']['loss'].append(loss.item())
        Logger()['train.csv']['accuracy'].append(accuracy.item())
        with torch.no_grad():
            Logger()['train.csv']['cls_w_mean'].append(classifier.cls_out.weight.mean().item())
            Logger()['train.csv']['cls_w_std'].append(classifier.cls_out.weight.std().item())
            Logger()['train.csv']['cls_w_min'].append(classifier.cls_out.weight.min().item())
            Logger()['train.csv']['cls_w_max'].append(classifier.cls_out.weight.max().item())

            Logger()['train.csv']['cls_b_mean'].append(classifier.cls_out.bias.mean().item())
            Logger()['train.csv']['cls_b_std'].append(classifier.cls_out.bias.std().item())
            Logger()['train.csv']['cls_b_min'].append(classifier.cls_out.bias.min().item())
            Logger()['train.csv']['cls_b_max'].append(classifier.cls_out.bias.max().item())

        if steps % 1000 == 0:
            Logger().log_all()

        if steps % args.skip_steps == 0:
            total_loss.append(loss.item())
            total_acc.append(accuracy.item())

            current_example, current_epoch = processor.get_train_progress()
            time_end = time.time()
            used_time = time_end - time_begin
            print('epoch: %d, progress: %d/%d, step: %d, ave loss: %f, ave acc: %f, speed: %f steps/s' %
                  (current_epoch, current_example, num_train_examples, steps,
                   np.mean(total_loss).item(), np.mean(total_acc).item(), args.skip_steps / used_time))

            total_loss, total_acc = [], []
            time_begin = time.time()




if __name__ == '__main__':
    main(args)