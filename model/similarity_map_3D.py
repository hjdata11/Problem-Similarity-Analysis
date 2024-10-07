
from absl import app, flags
from collections import Counter, OrderedDict

import torch as th
import pytorch_lightning as pl

import transformers

import pandas as pd
import numpy as np

import pickle

flags.DEFINE_string('input_file_path', '../data/PT15k/PT.pkl', '')
flags.DEFINE_string('preprocessed_file_path', '../data/PT15ktensor/', '')
flags.DEFINE_string('output_file_path', '../data/PT15ktensor/', '')
flags.DEFINE_string('checkpoint_path', '../data/checkpoint15k/', '')
flags.DEFINE_string('input_name', 'problem', '')
flags.DEFINE_string('label_name', 'type', '')
flags.DEFINE_integer('label_number', 56, '')
flags.DEFINE_string('model', 'bert-base-multilingual-cased', '')
flags.DEFINE_integer('batch_size', 3, '')

FLAGS = flags.FLAGS

# debugging
# import IPython ; IPython.embed() ; exit(1)

class PTDataSimilarityMap3D(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.prepare_data()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model, num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):

        def _count_items(l):
            # Create a counter object
            counts = Counter(l)
            # Sort by highest count first and place in ordered dictionary
            counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
            counts = OrderedDict(counts)
            return counts

        def _shrink_labels(labels):
            label_counts = _count_items(labels)
            label_to_include = list(label_counts.keys())[:FLAGS.label_number]

            label_idx_total = []
            tmp_labels = []

            for i, label in enumerate(labels):
                if label in label_to_include:
                    label_idx_total.append(i)
                    tmp_labels.append(label)

            shrinked_ints, shrinked_labels = pd.factorize(tmp_labels)
            return label_idx_total, shrinked_ints, shrinked_labels

        def _prepare_dataset():

            with open(FLAGS.input_file_path, 'rb') as f:
                self.df = pickle.load(f)

            self.label_idx_total, self.shrinked_ints, self.shrinked_labels= _shrink_labels(self.df[FLAGS.label_name])

            self.possible_labels = self.df[FLAGS.label_name].unique()
            # Number of unique labels
            self.num_labels = len(self.possible_labels)

            total_dataset = th.load(f'{FLAGS.preprocessed_file_path}PTtensorTotal.pt')

            return total_dataset

        self.total_dataset = _prepare_dataset()


    def forward(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pooled_output = th.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        pooled_output = pooled_output[:, 196, :]
        pooled_output = pooled_output.detach().cpu().numpy()
        return pooled_output

    def test_step(self, batch, batch_idx):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'pixel_values':     batch[2]}
        pooled_output = self.forward(inputs)
        return {'pooled_output': pooled_output}

    def test_epoch_end(self, outputs):

        def _save_data(problem_weights):
            
            # save new metadata
            df = self.df.loc[self.label_idx_total]

            for i, string in enumerate(df['problem']):
                new_string = ''.join(char for char in string if char.isalnum())
                df['problem'].iloc[i] = new_string

            df.to_csv(f'{FLAGS.output_file_path}3Dmetadata.tsv', index=False, sep='\t')

            # save vector
            problem_weights = problem_weights[self.label_idx_total]
            embedding_df = pd.DataFrame(problem_weights)

            embedding_df.to_csv(f'{FLAGS.output_file_path}3Doutput.tsv', sep='\t', index=None, header=None)

        pooled_output = [o['pooled_output'] for o in outputs]
        problem_weights = np.vstack(pooled_output)
        np.save(f'{FLAGS.output_file_path}PTweights.npy', problem_weights)
        _save_data(problem_weights)

    def test_dataloader(self):
        return th.utils.data.DataLoader(
                self.total_dataset,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=False,
                )

def main(_):
    model = PTDataSimilarityMap3D()
    trainer = pl.Trainer(
        default_root_dir = 'logs',
        gpus=(1 if th.cuda.is_available() else 0),
    )
    model = model.load_from_checkpoint(checkpoint_path=f'{FLAGS.checkpoint_path}last.ckpt')
    trainer.test(model)
    

if __name__ == '__main__':
    app.run(main)