import random
# import pprint
import scipy.misc
import numpy as np
# from time import gmtime, strftime
# from six.moves import xrange
import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
plt.switch_backend('agg')
import os, gzip

# from skimage import io
import cv2

import torch.utils.data as data

def load_Anime(dataset_filepath):
    tag_csv_filename = dataset_filepath.replace('images/', 'tags.csv')
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair',
                'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair',
                'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
                'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

    tag_csv = open(tag_csv_filename, 'r').readlines()
    # print(len(tag_csv))

    id_label = []
    for line in tag_csv:
        id, tags = line.split(',')  # id:1 tags:[orange hair,black eyes]
        label = np.zeros(len(tag_dict))

        for i in range(len(tag_dict)):
            if tag_dict[i] in tags:
                label[i] = 1   # [0,1,0,0,...,1,0]

        # Keep images with hair or eyes.
        if np.sum(label) == 2 or np.sum(label) == 1:
            id_label.append((id, label))  # tuple

    # Load file name of images.
    image_file_list = []
    for image_id, _ in id_label:
        image_file_list.append(image_id + '.jpg')

    # Resize image to 64x64.
    image_height = 64
    image_width = 64
    image_channel = 3

    # Allocate memory space of images and labels.
    images = np.zeros((len(image_file_list), image_width, image_height, image_channel))
    labels = np.zeros((len(image_file_list), len(tag_dict)))
    print('images.shape: ', images.shape)
    print('labels.shape: ', labels.shape)

    print('Loading images to numpy array...')
    data_dir = dataset_filepath
    for index, filename in enumerate(image_file_list):
        images[index] = cv2.cvtColor(
            cv2.resize(
                cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_COLOR),
                (image_width, image_height)),
            cv2.COLOR_BGR2RGB)
        labels[index] = id_label[index][1]

    print('Random shuffling images and labels...')
    np.random.seed(9487)
    indice = np.array(range(len(image_file_list)))
    np.random.shuffle(indice)
    images = images[indice]
    labels = labels[indice]

    print('[Tip 1] Normalize the images between -1 and 1.')
    # Tip 1. Normalize the inputs
    #   Normalize the images between -1 and 1.
    #   Tanh as the last layer of the generator output.
    return (images / 127.5) - 1, labels  # (0-255)/127.5 - 1 = (0-2)-1 = (-1,1)
    # return images / 255., labels
    # return (# of pictures, 64, 64, 3)  (# of pictures, 23)

class ImageFolder(data.Dataset):
    def __int__(self):
        super(ImageFolder, self).__int__()
        self.images, self.labels = load_Anime('../extra_data')

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
