
from absl import app, flags

import torch as th
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import pickle
from tqdm.notebook import tqdm
from PIL import Image
import numpy as np
from collections import defaultdict

flags.DEFINE_string('input_file_path', '../data/PT15k/PTTrainTest.pkl', '')
flags.DEFINE_string('output_file_path', '../data/PT15ktensor/', '')
flags.DEFINE_string('checkpoint_path', '../data/checkpoint15k', '')
flags.DEFINE_string('label_name', 'type', '')
flags.DEFINE_integer('epochs', 50, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('lr', 1e-5, '')
flags.DEFINE_float('eps', 1e-8, '')
flags.DEFINE_string('model', 'bert-base-multilingual-cased', '')
flags.DEFINE_integer('max_length', 256, '')

FLAGS = flags.FLAGS

# debugging
# import IPython ; IPython.embed() ; exit(1)

class BERTProblemTypeClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = transformers.ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.prepare_data()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model, num_labels=self.num_labels, output_attentions=False, output_hidden_states=False)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')
        self.best_acc = 0

    def load_checkpoint(self):
        old_weights = list(th.load(f'{FLAGS.checkpoint_path}/type_50/last.ckpt')['state_dict'].items())
        new_weights = self.model.state_dict()

        length = len(new_weights)
        i = 0
        for k, _ in new_weights.items():
            # if k == 'classifier.weight':
            #     break
            new_weights[k] = old_weights[i][1]
            i += 1

        self.model.load_state_dict(new_weights)

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)

        # Show Tokenized dataset
        def _show_tokenize(ids, problems):
            for id, problem in zip(ids, problems):
                token = tokenizer.tokenize(problem)
                print(problem)
                print(token)
                print(id)
                break
            return

        def _get_img(path):
            if path:
                return Image.open(path), [1] * 196
            else:
                return Image.new(mode="RGB", size=(224, 224)), [1] * 196

        def _tokenize(df):

            encoded_data_train = tokenizer.batch_encode_plus(df[df.data_type=='train'].problem.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=FLAGS.max_length, return_tensors='pt')
            encoded_data_test = tokenizer.batch_encode_plus(df[df.data_type=='test'].problem.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=FLAGS.max_length, return_tensors='pt')
            encoded_data_total = tokenizer.batch_encode_plus(df.problem.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=FLAGS.max_length, return_tensors='pt')
            
            #_show_tokenize(encoded_data_train['input_ids'], df[df.data_type=='train'].problem.values)

            imgs_train, imgs_mask_train, imgs_test, imgs_mask_test, imgs_total, imgs_mask_total = [], [], [], [], [], []
            for i, v in enumerate(df.data_type):
                img, mask = _get_img(df.iloc[i].image_path)
                if v == 'train':
                    imgs_train.append(img)
                    imgs_mask_train.append(mask)
                    imgs_total.append(img)
                    imgs_mask_total.append(mask)
                else:
                    imgs_test.append(img)
                    imgs_mask_test.append(mask)
                    imgs_total.append(img)
                    imgs_mask_total.append(mask)

            features_train = self.feature_extractor(imgs_train, return_tensors="pt")['pixel_values']
            features_test = self.feature_extractor(imgs_test, return_tensors="pt")['pixel_values']
            features_total = self.feature_extractor(imgs_total, return_tensors="pt")['pixel_values']
            
            imgs_mask_train = th.tensor(imgs_mask_train)
            imgs_mask_test = th.tensor(imgs_mask_test)
            imgs_mask_total = th.tensor(imgs_mask_total)

            input_ids_train = encoded_data_train['input_ids']
            attention_masks_train = encoded_data_train['attention_mask']
            attention_masks_train = th.cat([imgs_mask_train, attention_masks_train], dim=1)
            labels_train = th.tensor(df[df.data_type=='train'].label.values)

            input_ids_test = encoded_data_test['input_ids']
            attention_masks_test = encoded_data_test['attention_mask']
            attention_masks_test = th.cat([imgs_mask_test, attention_masks_test], dim=1)
            labels_test = th.tensor(df[df.data_type=='test'].label.values)

            input_ids_total = encoded_data_total['input_ids']
            attention_masks_total = encoded_data_total['attention_mask']
            attention_masks_total = th.cat([imgs_mask_total, attention_masks_total], dim=1)
            labels_total = th.tensor(df.label.values)

            train_dataset = TensorDataset(input_ids_train, attention_masks_train, features_train, labels_train)
            test_dataset = TensorDataset(input_ids_test, attention_masks_test, features_test, labels_test)
            total_dataset = TensorDataset(input_ids_total, attention_masks_total, features_total, labels_total)

            return train_dataset, test_dataset, total_dataset

        def _prepare_dataset():

            with open(FLAGS.input_file_path, 'rb') as f:
                df = pickle.load(f)

            self.possible_labels = df[FLAGS.label_name].unique()
            # Number of unique labels
            self.num_labels = len(self.possible_labels)

            train_dataset, test_dataset, total_dataset = _tokenize(df)

            th.save(train_dataset, f'{FLAGS.output_file_path}PTtensorTrain.pt')
            th.save(test_dataset, f'{FLAGS.output_file_path}PTtensorTest.pt')
            th.save(total_dataset, f'{FLAGS.output_file_path}PTtensorTotal.pt')
            th.save(self.possible_labels, f'{FLAGS.output_file_path}Possible_Labels.pt')

            return train_dataset, test_dataset

        self.train_dataset, self.test_dataset = _prepare_dataset()

    def tmp_prepare_dataset(self):
        self.train_dataset = th.load(f'{FLAGS.output_file_path}PTtensorTrain.pt')
        self.test_dataset = th.load(f'{FLAGS.output_file_path}PTtensorTest.pt')
        self.possible_labels = th.load(f'{FLAGS.output_file_path}Possible_Labels.pt')
        self.num_labels = len(self.possible_labels)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'pixel_values':     batch[2]}
        logits = self.forward(inputs)
        loss = self.loss(logits, batch[3]).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'pixel_values':     batch[2]}
        logits = self.forward(inputs)
        loss = self.loss(logits, batch[3])
        acc = (logits.argmax(-1) == batch[3]).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}

        if self.best_acc < acc:
            self.best_acc = acc
            th.save(self.model.state_dict(), f'{FLAGS.checkpoint_path}/best.ckpt')

        tqdm.write(f'Validation loss: {loss}')
        tqdm.write(f'Accuracy: {acc}')
        return {**out, 'log': out}
        
    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_dataset,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_dataset,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=True,
                )

    def configure_optimizers(self):
        optimizer =  th.optim.AdamW(self.parameters(), lr=FLAGS.lr, eps=FLAGS.eps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(self.train_dataset)*FLAGS.epochs)
        return [optimizer], [scheduler]

def main(_):
    model = BERTProblemTypeClassifier()
    model.load_checkpoint()
    checkpoint_callback = ModelCheckpoint(dirpath=FLAGS.checkpoint_path, save_last=True, filename='{epoch:03d}')
    trainer = pl.Trainer(
        default_root_dir = 'logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)

if __name__ == '__main__':
    app.run(main)
