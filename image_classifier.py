import os
import time
import numpy as np
import logging
import caffe
from bing_cluster import bing_cluster
import pandas as pd
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

    default_args['bing_model_file']='{}/models/bing/ObjNessB2W8MAXBGR'.format(PROJECT_DIRNAME);
    default_args['image_dim']=227
    default_args['raw_scale']=255
    default_args['gpu_mode']= True
    default_args['context_pad']=16
    default_args['cluster_num']=10
    default_args['top_k']=20
    default_args['max_ratio']=4
    default_args['min_size']=100
    def __init__(self,model_def_file=default_args['model_def_file'],
            pretrained_model_file=default_args['pretrained_model_file'],
            mean_file=default_args['mean_file'],
            raw_scale=default_args['raw_scale'],
            image_dim=default_args['image_dim'],
            gpu_mode=default_args['gpu_mode'],
            bing_model_file=default_args['bing_model_file'],
            context_pad=default_args['context_pad'],
            cluster_num=default_args['cluster_num'],
            top_k=default_args['top_k'],
            max_ratio=default_args['max_ratio'],
            min_size=default_args['min_size']
            ):
        logging.info('Loading net and associated files...')
        self.net = caffe.Classifier(
                model_def_file,pretrained_model_file,
                image_dims=(image_dim,image_dim),raw_scale=raw_scale,
                mean=np.load(mean_file),channel_swap=(2,1,0),gpu=gpu_mode)
        self.bing = bing_cluster(cluster_num,top_k,max_ratio,min_size)
        self.configure_crop(context_pad)
        self.bing.load_bing_model(bing_model_file)
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
        boxes = self.bing.get_bing_of_image(image_filename)
        boxes = self.bing.cluster_boxes(boxes)
        image = caffe.io.load_image(image_filename).astype(np.float32)
        windows_inputs = []
        for index , row in boxes.iterrows():
            windows_inputs.append(self.crop(image,row))
        
        caffe_in = np.zeros((len(windows_inputs),windows_inputs[0].shape[2])+
                self.net.blobs[self.net.inputs[0]].data.shape[2:],
                dtype=np.float32)
        for index , window_in in enumerate(windows_inputs):
            caffe_in[index] = self.net.preprocess(self.net.inputs[0],window_in)
        out = self.net.forward_all(blobs=['fc7'], **{self.net.inputs[0]: caffe_in})
        fc7_feature = out['fc7'].squeeze(axis=(2,3))
        #print fc7_feature.shape
        endtime = time.time()
        logging.info("One image bing cnn feature spend {:.3f}".format(endtime-starttime))
        return fc7_feature 
    def crop(self,im,row):
        """
        Crop a window from the image for detection. Include surrounding context 
        arrording to the `context_pad` configuration.

        Take
        im:H x W x K image ndarray to crop
        row: bounding box coordinates as ymin , xmin , ymax , xmax

        Give 
        crop: cropped window.
        """
        window=np.array((row["ymin"],row["xmin"],row["ymax"],row["xmax"]))
        crop = im[window[0]:window[2], window[1]:window[3]]
        if self.context_pad:
            box = window.copy()
            crop_size = self.net.blobs[self.net.inputs[0]].width
            scale = crop_size / (1. * crop_size - self.context_pad * 2 )
            # Crop a box + surrounding context
            half_h = (box[2] - box[0] + 1) / 2.
            half_w = (box[3] - box[1] + 1) / 2.
            center = (box[0] + half_h , box[1] + half_w)
            scaled_dims = scale * np.array((-half_h,-half_w,half_h,half_w))
            box = np.round(np.tile(center,2) + scaled_dims )
            full_h = box[2] - box[0] + 1
            full_w = box[3] - box[1] + 1
            scale_h = crop_size / full_h
            scale_w = crop_size / full_w
            pad_y = round(max(0, -box[0]) * scale_h) # amount out-of-bounds
            pad_x = round(max(0, -box[1]) * scale_w)
            # Clip box to image dimensions
            im_h, im_w = im.shape[:2]
            box = np.clip(box, 0., [im_h, im_w, im_h, im_w])
            clip_h = box[2] - box[0] + 1
            clip_w = box[3] - box[1] + 1
            assert(clip_h > 0 and clip_w > 0)
            crop_h = round(clip_h * scale_h)
            crop_w = round(clip_w * scale_w)
            if pad_y + crop_h > crop_size:
                crop_h = crop_size - pad_y
            if pad_x + crop_w > crop_size:
                crop_w = crop_size - pad_x
            # collect with context padding and place in input
            # with mean padding
            context_crop = im[box[0]:box[2], box[1]:box[3]]
            context_crop = caffe.io.resize_image(context_crop, (crop_h, crop_w))
            crop = self.crop_mean.copy()
            crop[pad_y:(pad_y + crop_h), pad_x:(pad_x + crop_w)] = context_crop
        return crop
    def configure_crop(self,context_pad):
        """
        Configure amount of context for cropping.
        If context is included, make the special input mean for context padding

        Take 
        context_pad: amount of context for cropping
        """
        self.context_pad = context_pad
        if self.context_pad:
            raw_scale = self.net.raw_scale.get(self.net.inputs[0])
            channel_order = self.net.channel_swap.get(self.net.inputs[0])
            mean = self.net.mean.get(self.net.inputs[0])
            if mean is not None:
                crop_mean = mean.copy().transpose((1,2,0))
                if channel_order is not None:
                    channel_order_inverse = [channel_order.index(i)
                            for i in range(crop_mean.shape[2])]
                    crop_mean = crop_mean[:,:, channel_order_inverse]
                if raw_scale is not None:
                    crop_mean /= raw_scale
                self.crop_mean = crop_mean
            else:
                self.crop_mean = np.zeros(self.net.blobs[self.net.inputs[0]].data.shape,dtype=np.float32)
