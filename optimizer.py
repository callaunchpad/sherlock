## optimizer.py 

import mxnet as mx
import json
import os
import logging

def read_data_params(self, config):
	''' reads json file config and sets up parameters '''
	self.logger.info("reading data params")
	with open(config) as data_file:
		self.cfg = json.load(data_file)

	self.dataset_dir = self.cfg['dataset_dir']
	self.num_epoch = self.cfg['num_epoch']
	self.learning_rate = self.cfg['learning_rate']
	self.train_batch_size = self.cfg['train_batch_size']
	self.val_batch_size = self.cfg['val_batch_size']
	self.im_width = self.cfg['im_width']
	self.im_height = self.cfg['im_height']
	self.im_channels = self.cfg['im_channels']


def setup(self):
	''' main setup function '''
	self.logger.info("beginning setup")
	self.setup_data_iters()

def setup_data_iters(self):
	''' sets up the training and validation iterators '''
	self.logger.info("setting up data iterators")
	self.train_iter = mx.io.ImageRecordIter(
    # Dataset Parameter, indicating the data file, please check the data is already there
    path_imgrec=os.path.join(self.dataset_dir, 'train.rec'),
    # Dataset Parameter, indicating the image size after preprocessing
    data_shape=(self.im_channels, self.im_width, self.im_height),
    # Batch Parameter, tells how many images in a batch
    batch_size=self.train_batch_size,
    # Augmentation Parameter, randomly shuffle the data
    shuffle=True,
    # Backend Parameter, preprocessing thread number
    preprocess_threads=4,
    # Backend Parameter, prefetch buffer size
    prefetch_buffer=1)

    self.val_iter = mx.io.ImageRecordIter(
    # Dataset Parameter, indicating the data file, please check the data is already there
    path_imgrec=os.path.join(self.dataset_dir, 'val.rec'),
    # Dataset Parameter, indicating the image size after preprocessing
    data_shape=(self.im_channels, self.im_width, self.im_height),
    # Batch Parameter, tells how many images in a batch
    batch_size=self.train_batch_size,
    # Augmentation Parameter, randomly shuffle the data
    shuffle=True,
    # Backend Parameter, preprocessing thread number
    preprocess_threads=4,
    # Backend Parameter, prefetch buffer size
    prefetch_buffer=1)

def __main__():
	''' main run loop '''
	self.logger = logging.getLogger('optimizer')
	self.logger.setLevel(logging.INFO)
	self.logger.info("beginning setup")
	self.setup()

