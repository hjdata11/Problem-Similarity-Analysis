
from absl import app, flags

import torch as th
import pytorch_lightning as pl

import transformers

import numpy as np

from tqdm.notebook import tqdm

flags.DEFINE_string('input_file_path', '../data/PT15ktensor/', '')
flags.DEFINE_string('checkpoint_path', '../data/checkpoint15k/', '')
flags.DEFINE_string('model', 'bert-base-multilingual-cased', '')
flags.DEFINE_integer('batch_size', 3, '')

FLAGS = flags.FLAGS

# debugging
# import IPython ; IPython.embed() ; exit(1)

class PTDataPostprocessing(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.prepare_data()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model, num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):

        def _label_dict(possible_labels):
            label_dict = {}
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            return label_dict

        self.test_dataset = th.load(f'{FLAGS.input_file_path}PTtensorTest.pt')
        self.possible_labels = th.load(f'{FLAGS.input_file_path}Possible_Labels.pt')
        # Number of unique labels
        self.num_labels = len(self.possible_labels)
        self.label_dict = _label_dict(self.possible_labels)
        return

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs.logits

    def test_step(self, batch, batch_idx):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'pixel_values':     batch[2]}
        logits = self.forward(inputs)
        loss = self.loss(logits, batch[3])
        acc = (logits.argmax(-1) == batch[3]).float()
        logits = logits.detach().cpu().numpy()
        labels = batch[3].cpu().numpy()
        return {'loss': loss, 'acc': acc, 'logits': logits, 'labels': labels}

    def test_epoch_end(self, outputs):

        def _accuracy_per_class(preds, labels):
            label_dict_inverse = {v: k for k, v in self.label_dict.items()}

            preds_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()

            for label in np.unique(labels_flat):
                y_preds = preds_flat[labels_flat==label]
                y_true = labels_flat[labels_flat==label]
                tqdm.write(f'Class: {label_dict_inverse[label]}')
                tqdm.write(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

        predictions = [o['logits'] for o in outputs]
        predictions = np.concatenate(predictions, axis=0)
        true_vals = [o['labels'] for o in outputs]
        true_vals = np.concatenate(true_vals, axis=0)
        _accuracy_per_class(predictions, true_vals)

        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        tqdm.write(f'Accuracy: {acc}\n')

        out = {'test_loss': loss, 'test_acc': acc}

        return {**out, 'log': out}

    def test_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_dataset,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=True,
                )

def main(_):
    model = PTDataPostprocessing()
    trainer = pl.Trainer(
        default_root_dir = 'logs',
        gpus=(1 if th.cuda.is_available() else 0),
    )
    model = model.load_from_checkpoint(checkpoint_path=f'{FLAGS.checkpoint_path}last.ckpt')
    trainer.test(model)
    

if __name__ == '__main__':
    app.run(main)
