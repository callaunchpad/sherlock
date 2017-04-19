'''
authors: vishal satish, darren lee 
hello peter 
'''

import mxnet as mx
import json
import os
import logging


def read_data_params(self, config):
	''' reads json file config and sets up parameters '''
	# TODO: generalize to any dictionary input, not just json
	self.logger.info("reading data params")
	with open(config) as data_file:
		self.cfg = json.load(data_file)

	self.architecture = self.cfg['architecture']

	self.dataset_dir = self.cfg['dataset_dir']
	self.im_width = self.cfg['im_width']
	self.im_height = self.cfg['im_height']
	self.im_channels = self.cfg['im_channels']

	self.train_batch_size = self.cfg['train_batch_size']
	self.val_batch_size = self.cfg['val_batch_size']

	self.num_gpus = self.cfg['num_gpus']
	self.num_epoch = self.cfg['num_epoch']
	self.learning_rate = self.cfg['learning_rate']
	self.optimizer = self.cfg['optimizer']
	self.train_log_frequency = self.cfg['train_log_frequency']
	self.eval_metric = self.eval_metric['eval_metric']

	self.conv1_num_filter = self.architecture['conv1']['num_filter']
	self.conv1_filter_size = self.architecture['conv1']['filter_size']
	self.conv1_pool_filter_size = self.architecture['conv1']['pool_filter_size']
	self.conv1_act_type = self.architecture['conv1']['act_type']
	self.conv1_pool_type = self.architecture['conv1']['pool_type']
	self.conv1_pool_stride = self.architecture['conv1']['pool_stride']

	self.conv2_num_filter = self.architecture['conv2']['num_filter']
	self.conv2_filter_size = self.architecture['conv2']['filter_size']
	self.conv2_pool_filter_size = self.architecture['conv2']['pool_filter_size']
	self.conv2_act_type = self.architecture['conv2']['act_type']
	self.conv2_pool_type = self.architecture['conv2']['pool_type']
	self.conv2_pool_stride = self.architecture['conv2']['pool_stride']

	self.fc1_num_hidden = self.architecture['fc1']['_num_hidden']
	self.fc1_act_type = self.architecture['fc1']['_act_type']

	self.fc2_num_hidden = self.architecture['fc2']['_num_hidden']


def setup(self, config):
	''' main setup function '''
	self.logger.info("beginning setup")
	self.read_data_params(config)
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
    batch_size=self.val_batch_size,
    # Augmentation Parameter, randomly shuffle the data
    shuffle=True,
    # Backend Parameter, preprocessing thread number
    preprocess_threads=4,
    # Backend Parameter, prefetch buffer size
    prefetch_buffer=1)

def build_square_filter(self, size):
	return (size, size)

def build_network(self, input_im_node):
	# conv1
	conv1 = buildConvolution(input_node=input_im_node, 
		num_filter=self.conv1_num_filter, 
		conv_kernel=self.build_square_filter(self.conv1_filter_size), 
		pool_kernel=self.build_square_filter(self.conv1_pool_filter_size), 
		act_type=self.conv1_act_type,
		pool_type=self.conv1_pool_type, 
		pool_stride=self.build_square_filter(self.conv1_pool_stride), 
		name='1')

	# conv2 
	conv2 = buildConvolution(input_node=conv1, 
		num_filter=self.conv2_num_filter, 
		conv_kernel=self.build_square_filter(self.conv2_filter_size), 
		pool_kernel=self.build_square_filter(self.conv2_pool_filter_size), 
		act_type=self.conv2_act_type,
		pool_type=self.conv2_pool_type, 
		pool_stride=self.build_square_filter(self.conv2_pool_stride), 
		name='2')

	# first fullc layer 
	flatten = mx.sym.Flatten(data=conv2)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=self.fc1_num_hidden)
	act3 = mx.symbol.Activation(data=fc1, act_type=self.fc1_act_type)

	# second fullc layer
	fc2 = mx.sym.FullyConnected(data=act3, num_hidden=self.fc2_num_hidden) # 101

	# softmax loss
	self.symbol = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

def buildConvolution(input_node, num_filter, conv_kernel, pool_kernel, conv_stride=(1,1), 
	pad=(0, 0), apply_norm=False, act_type='relu', pool_type='max', 
	pool_stride=(1,1), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=input_node, 
    	num_filter=num_filter, 
    	kernel=conv_kernel, 
    	stride=conv_stride, 
    	pad=pad, 
    	name='conv_%s%s' %(name, suffix))
    if apply_norm:
    	bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    	act = mx.symbol.Activation(data=bn, 
    		act_type=act_type, 
    		name='%s_%s%s' %(act_type, name, suffix))
    else:
    	act = mx.symbol.Activation(data=conv, 
    		act_type=act_type, 
    		name='%s_%s%s' %(act_type, name, suffix))
    pool = mx.sym.Pooling(data=act, 
    	pool_type=pool_type, 
		kernel=pool_kernel, 
		stride=pool_stride)
    return pool

def optimize(self):
	# TODO: add more params to self.model
	self.model = mx.model.FeedForward(
		ctx=[mx.gpu(int(i)) for i in range(self.num_gpus)], 
		symbol=self.symbol, 
		num_epoch=self.num_epoch, 
		learning_rate=self.learning_rate, 
		optimizer=self.optimizer)
	self.model.fit(
		X=self.train_iter, 
		eval_data=self.val_iter, 
		batch_end_callback=mx.callback.Speedometer(
			self.train_batch_size, 
			self.train_log_frequency),
		eval_metric=self.eval_metric)
	logger.info("final validation accuracy: %s" %(self.model.score(self.val_iter)))

def cleanup(self):
	# TODO: cleanup here

def __main__():
	''' main run loop '''
	self.logger = logging.getLogger('optimizer')
	self.logger.setLevel(logging.INFO)
	self.logger.info("beginning setup")
	self.setup('config.json') #TODO: json file 
	self.build_network()
	self.optimize()
	self.cleanup()

