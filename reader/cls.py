import tokenization
import numpy as np
import csv
import os


class MnliDataProcessor:
    def __init__(self,
                 data_dir,
                 vocab_path,
                 max_seq_len,
                 do_lower_case,
                 random_seed=None):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case
        )
        self.vocab = self.tokenizer.vocab

        if random_seed is not None:
            np.random.seed(random_seed)

        self.current_train_example = -1
        self.current_train_epoch = -1
        self.num_examples = {'train': -1, 'dev': -1}

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        '''Reads a tab separated value file.'''
        with open(input_file, 'r') as f:
            lines = list(csv.reader(f, delimiter='\t', quotechar=quotechar))
        return lines

    def _create_examples(self, lines, set_type):
        '''Creates examples for the training and dev sets.'''
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = '%s-%s' % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append({
                'guid': guid,
                'text_a': text_a,
                'text_b': text_b,
                'label': label
            })
        return examples

    def _convert_example(self, example):
        label_map = {}
        for (i, label) in enumerate(self.get_labels()):
            label_map[label] = i

        tokens_a = self.tokenizer.tokenize(example['text_a'])
        tokens_b = self.tokenizer.tokenize(example['text_b'])

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "-3"

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        max_length = self.max_seq_len - 3
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

        # The convention in BERT is:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the word piece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_pos = list(range(len(input_ids)))
        label_id = label_map[example['label']]

        return [input_ids, segment_ids, input_pos, label_id]

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'dev_matched.tsv')), 'dev_matched')

    def get_labels(self):
        '''Gets the list of labels for MNLI data set.'''
        return ['contradiction', 'entailment', 'neutral']

    def generate_batch_data(self, batch_data):
        pad_id = self.vocab['[PAD]']
        num_buckets = 1
        max_seq_len = 128

        batch_src_ids = [inst[0] for inst in batch_data]
        batch_sent_ids = [inst[1] for inst in batch_data]
        batch_pos_ids = [inst[2] for inst in batch_data]
        labels = [inst[3] for inst in batch_data]
        labels = np.array(labels).astype(np.int64).reshape([-1, 1])

        # Bucketing
        bin_width = max_seq_len / num_buckets
        boundaries = np.linspace(bin_width, max_seq_len, num_buckets, dtype=np.int32).tolist()
        max_len_actual = max(len(inst) for inst in batch_data)
        if max_len_actual == max_seq_len:
            max_len = max_seq_len
        else:
            max_len = int(boundaries[int(max_len_actual / bin_width)])
        # max_len = max(len(inst) for inst in insts)

        src_id = np.array([inst + [pad_id] * (max_len - len(inst)) for inst in batch_src_ids])
        # src_id = src_id.astype(np.int64).reshape([-1, max_len, 1])

        # This is used to avoid attention on paddings.
        self_input_mask = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in batch_src_ids])
        self_input_mask = np.expand_dims(self_input_mask, axis=-1)

        pos_id = np.array([inst + [pad_id] * (max_len - len(inst)) for inst in batch_pos_ids])
        # pos_id = pos_id.astype(np.int64).reshape([-1, max_len, 1])

        sent_id = np.array([inst + [pad_id] * (max_len - len(inst)) for inst in batch_sent_ids])
        # sent_id = sent_id.astype(np.int64).reshape([-1, max_len, 1])

        return [src_id, pos_id, sent_id, self_input_mask, labels]

    def get_num_examples(self, phase):
        '''Get number of examples for train or dev.'''
        if phase not in ['train', 'dev']:
            raise ValueError(
                'Unknown phase, which shold be in [\'train\', \'dev\'].'
            )
        return self.num_examples[phase]

    def get_train_progress(self):
        '''Gets progress for training phase.'''
        return self.current_train_example, self.current_train_epoch

    def data_generator(self, batch_size, phase='train', epoch=1, shuffle=True):
        '''
        Generates data for train and dev.

        :param batch_size: int. The batch size of generated data.
        :param phase: string. The phase for which to generate data.
        :param epoch: int. Total epochs to generate data.
        :param shuffle: bool. Whether to shuffle examples.
        '''
        if phase == 'train':
            examples = self.get_train_examples(self.data_dir)
            self.num_examples['train'] = len(examples)
        elif phase == 'dev':
            examples = self.get_dev_examples(self.data_dir)
            self.num_examples['dev'] = len(examples)
        else:
            raise ValueError(
                'Unknown phase, which should be in [\'train\', \'dev\'].'
            )

        def instance_reader():
            for epoch_index in range(epoch):
                if shuffle:
                    np.random.shuffle(examples)
                if phase == 'train':
                    self.current_train_epoch = epoch_index
                for (index, example) in enumerate(examples):
                    if phase == 'train':
                        self.current_train_example = index + 1
                    yield self._convert_example(example)

        def batch_reader(reader, batch_size):
            batch, total_token_num = [], 0
            for example in reader():
                token_ids = example[0]
                if len(batch) < batch_size:
                    batch.append(example)
                    total_token_num += len(token_ids)
                else:
                    yield batch, total_token_num
                    batch, total_token_num = [example], len(token_ids)

            if len(batch) > 0:
                yield batch, total_token_num

        def wrapper():
            for batch_data, total_token_num in batch_reader(instance_reader, batch_size):
                batch_data = self.generate_batch_data(batch_data)
                yield batch_data

        return wrapper


if __name__ == '__main__':
    print('Create processor')
    processor = MnliDataProcessor(
        data_dir='/home/cvds_lab/maxim/dataset_for_bert/GLUE/glue_data/MNLI',
        vocab_path='/home/cvds_lab/maxim/dataset_for_bert/ckps/uncased_L-24_H-1024_A-16/vocab.txt',
        max_seq_len=128,
        do_lower_case=True,
        random_seed=1
    )

    print('Get num_labels')
    num_labels = len(processor.get_labels())

    print('Create data_generator')
    train_data_generator = processor.data_generator(
        batch_size=16,
        phase='train',
        epoch=3,
        shuffle=True
    )

    print('Get num_train_examples')
    num_train_examples = processor.get_num_examples(phase='train')

    print(num_labels)
    print(num_train_examples)
    data = next(train_data_generator())

    # import pickle
    #
    #
    # def save_pickle(data, fname):
    #     with open(fname, 'wb') as f:
    #         pickle.dump(data, f)
    #
    # save_pickle(data, '/home/cvds_lab/maxim/transformer_investigation/notebooks/ckp/my_input.pkl')

