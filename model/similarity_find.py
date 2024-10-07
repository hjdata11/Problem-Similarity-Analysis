
import argparse

import numpy as np

import os
import pickle
from numpy import dot
from numpy.linalg import norm
import re
from PIL import Image

parser = argparse.ArgumentParser(description='retrieval')
parser.add_argument('--input_file_path', default='../data/PT15k/PT.pkl', type=str, help='input_file_path')
parser.add_argument('--preprocessed_file_path', default='../data/PT15ktensor/', type=str, help='preprocessed_file_path')
parser.add_argument('--result_img_path', default='../data/imgs_retrieval/', type=str, help='result_image_path')
parser.add_argument('--input_name', default='problem', type=str, help='problem name string')
parser.add_argument('--label_name', default='type', type=str, help='label')
parser.add_argument('--image_path', default='image_path', type=str, help='image_path')
parser.add_argument('--retrieval_num', default=10, type=int, help='retrieval_num')
parser.add_argument('--query_index', default=5, type=int, help='query index')
opt = parser.parse_args()
input_file_path, preprocessed_file_path, result_img_path, input_name, label_name, image_path, retrieval_num, query_index = opt.input_file_path, opt.preprocessed_file_path, opt.result_img_path, opt.input_name, opt.label_name, opt.image_path, opt.retrieval_num, opt.query_index

# debugging
# import IPython ; IPython.embed() ; exit(1)

class PTDataSimilarityFind():
    def __init__(self):
        super().__init__()
        self.prepare_data()

    def prepare_data(self):

        def _prepare_dataset():

            with open(input_file_path, 'rb') as f:
                self.df = pickle.load(f)

            indexes = list(range(0, len(self.df)))
            latent_features = np.load(f'{preprocessed_file_path}PTweights.npy')
            index_dict = {'indexes':indexes,'features':latent_features}
            problems = self.df[input_name]
            labels = self.df[label_name]
            image_paths = self.df[image_path]

            return latent_features, index_dict, problems, labels, image_paths
            
        self.latent_features, self.index_dict, self.problems, self.labels, self.image_paths = _prepare_dataset()

    def process (self):

        def _createFolder(directory):
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            except OSError:
                    print('Error: Createing directory. ' + directory)

        def _euclidean(a, b):
            return np.linalg.norm(a - b)

        def _cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

        def _jaccard_similarity(list1, list2):
            intersection = len(list(set(list1).intersection(list2)))
            union = (len(set(list1)) + len(set(list2))) - intersection
            return float(intersection) / union

        def _perform_search(queryFeatures, index, retrieval_num=10):
            
            results = []

            for i in range(0, len(index["features"])):
                d = _euclidean(queryFeatures, index["features"][i])
                results.append((d, i))
    
            results = sorted(results)[:retrieval_num]
            return results
        
        queryFeatures = self.latent_features[query_index]
        results = _perform_search(queryFeatures, self.index_dict, retrieval_num=retrieval_num)

        print(f"선택된 문제 유형 : {self.labels[query_index]}")
        print(f"선택된 문제 : {self.problems[query_index]}")
        print()
        folder_path = f'{result_img_path}{query_index}'
        _createFolder(folder_path)
        if self.image_paths[query_index]:
            query_image_name = re.split('[/.]', self.image_paths[query_index])[-2]
            Image.open(self.image_paths[query_index]).save(f'{folder_path}/0_query_{query_image_name}.jpg')
        
        print("선택된 문제와 가까운 문제")
        for idx, (d, j) in enumerate(results):
            print(f'순서 : {idx+1}')
            print(f'문제 유형 : {self.labels[j]}')
            print(f'문제 번호 : {self.problems[j]}')
            print()
            if self.image_paths[j]:
                retrieval_image = Image.open(self.image_paths[j]).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
                image_name = re.split('[/.]', self.image_paths[j])[-2]
                retrieval_image.save(f'{folder_path}/{idx+1}_{image_name}.jpg')


if __name__ == '__main__':
    model = PTDataSimilarityFind()
    model.process()
