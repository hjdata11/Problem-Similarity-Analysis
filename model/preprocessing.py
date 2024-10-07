
from absl import app, flags
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import os
import re

flags.DEFINE_string('input_file_path', '../data/problems15k.pkl', '')
flags.DEFINE_string('output_file_path', '../data/PT15k/', '')
flags.DEFINE_integer('number_of_types', 10, '')
flags.DEFINE_string('curriculum', 'curriculum', '')
flags.DEFINE_string('form', 'form', '')
flags.DEFINE_string('difficulty', 'difficulty', '')
flags.DEFINE_string('type', 'type', '')
flags.DEFINE_string('input_name', 'problem', '')
flags.DEFINE_string('label_name', 'type', '')
flags.DEFINE_float('test_size', 0.1, '')
flags.DEFINE_integer('random_state', 2020, '')
flags.DEFINE_string('image_path', '../data/256_images/', '')

FLAGS = flags.FLAGS

# debugging
# import IPython ; IPython.embed() ; exit(1)

class PTDataPreprocessing():
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        with open(FLAGS.input_file_path, 'rb') as f:
            dataset = pickle.load(f)

        def _extract_items(dataset):
            inputs, types, curriculums, forms, difficulties, pairs, img_paths = [], [], [], [], [], [], []

            image_file_paths = [f for f in os.listdir(FLAGS.image_path) if os.path.isfile(os.path.join(FLAGS.image_path, f))]
            
            for data in dataset:
                inputs.append(data[2] + " " + data[3]) # Problem
                types.append(data[1]) # Type
                curriculums.append(data[4]) # Curriculums
                #forms.append(data[5]) # Form
                #difficulties.append(data[6]) # Difficulty

                img_path = ""
                file = str(data[0]) + ".jpg"
                if os.path.isfile(os.path.join(FLAGS.image_path, file)):
                    img_path = os.path.join(FLAGS.image_path, str(data[0]) + ".jpg")
                img_paths.append(img_path)

                pairs.append([data[2] + " " + data[3], data[1], data[4], img_path])#, data[5], data[6]]) # type, curriculum, form, difficulty, image_path
            return inputs, types, curriculums, forms, difficulties, img_paths, pairs

        def _sort_items(labels):
            counts = Counter(labels)
            # Sort by highest count first and place in ordered dictionary
            counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
            counts = OrderedDict(counts)
            return counts

        def _filter_items(dataset, types):
            counts = Counter(types)
            # Filter dataset more than number of 10 in label data
            filtered_types = [x[0] for x in counts.items() if x[1] >= FLAGS.number_of_types]
            dataset = [data for data in dataset if data[1] in filtered_types]
            return dataset

        def _create_train_test(df):
            possible_labels = df[FLAGS.label_name].unique()
            # Create number of labels
            label_dict = {}
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            df['label'] = df[FLAGS.label_name].replace(label_dict)

            X_train, X_test, y_train, y_test = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=FLAGS.test_size, 
                                                  random_state=FLAGS.random_state, 
                                                  stratify=df.label.values)

            df['data_type'] = ['not_set']*df.shape[0]
            df.loc[X_train, 'data_type'] = 'train'
            df.loc[X_test, 'data_type'] = 'test'
            df.groupby([FLAGS.label_name, 'label', 'data_type']).count()
            return df


        inputs, types, curriculums, forms, difficulties, img_paths, pairs = _extract_items(dataset)
        dataset = _filter_items(dataset, types)
        inputs, types, curriculums, forms, difficulties, img_paths, pairs = _extract_items(dataset)

        print(f'{len(Counter(types))}종류의 Label 존재')

        df = pd.DataFrame(pairs, columns=[FLAGS.input_name, FLAGS.type, FLAGS.curriculum, "image_path"])#, FLAGS.form,  FLAGS.difficulty])

        # Create input & label dataset
        with open(f'{FLAGS.output_file_path}PT.pkl', "wb") as file:
            pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

        df = _create_train_test(df)

        # Create train & test dataset
        with open(f'{FLAGS.output_file_path}PTTrainTest.pkl', "wb") as file:
            pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)


def main(_):
    preprocessing = PTDataPreprocessing()
    # Create dataset
    preprocessing.prepare_data()


if __name__ == '__main__':
    app.run(main)
