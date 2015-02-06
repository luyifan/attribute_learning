import os
import time
import numpy as np
import logging
import caffe
PROJECT_DIRNAME = os.path.abspath(os.path.dirname(__file__))
logging.getLogger().setLevel(logging.INFO)
class image_classifier:
    default_args = {
            'model_def_file' : ('{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(PROJECT_DIRNAME)),
            'pretrained_model_file' : ('{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(PROJECT_DIRNAME)),
            'mean_file':('{}/models/ilsvrc12/ilsvrc_2012_mean.npy'.format(PROJECT_DIRNAME)),
            }
    for key , val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                    "File for {} is missing. Should be at:{}".format(key,val))

    default_args['image_dim']=227
    default_args['raw_scale']=255
    default_args['gpu_mode']= True
    def __init__(self,model_def_file=default_args['model_def_file'],
            pretrained_model_file=default_args['pretrained_model_file'],
            mean_file=default_args['mean_file'],
            raw_scale=default_args['raw_scale'],
            image_dim=default_args['image_dim'],
            gpu_mode=default_args['gpu_mode']):
        logging.info('Loading net and associated files...')
        self.net = caffe.Classifier(
                model_def_file,pretrained_model_file,
                image_dims=(image_dim,image_dim),raw_scale=raw_scale,
                mean=np.load(mean_file),channel_swap=(2,1,0),gpu=gpu_mode)
    def get_cnn_feature_of_image(self,image_filename):
        starttime = time.time()
        one_image = caffe.io.load_image(image_filename)
        cnn_feature=self.net.fc7_feature([one_image]) 
        #print cnn_feature.shape
        endtime = time.time()
        logging.info("One image cnn feature spend {:.3f}".format(endtime-starttime))
        return cnn_feature
    def get_bing_cnn_feature_of_image(self,image_filename):
        starttime = time.time()
        
        endtime = time.time()
        logging.info("One image bing cnn feature spend {:.3f}".format(endtime-starttime))
        return 
