from bing import Bing
from bing import Boxes
import pandas as pd
import os
import logging
import time
import numpy as np
from sklearn import cluster

PROJECT_DIRNAME = os.path.abspath(os.path.dirname(__file__))
MODEL_DIRNAME = os.path.join(PROJECT_DIRNAME,"models/bing")
DEFAULT_MODEL_FILE = os.path.join(MODEL_DIRNAME,"ObjNessB2W8MAXBGR")
logging.getLogger().setLevel(logging.INFO)
class bing_cluster:
    def __init__(self,cluster_num=10,top_k=10,max_ratio=4,min_size=100):
        logging.info("Init the bing and cluster parameter")
        self.cluster_num = cluster_num
        self.top_k = top_k
        self.max_ratio = max_ratio
        self.min_size = min_size
        self.spectral = cluster.SpectralClustering(n_clusters=self.cluster_num,affinity='precomputed') 
    def load_bing_model(self,model_file=DEFAULT_MODEL_FILE):
        logging.info("Load Bing Model ...")
        self.bing = Bing(2,8,2)
        self.bing.loadTrainModel(model_file)
    def get_bing_of_image(self,image_filename,numPerSz=130):
        boxes = self.bing.getBoxesOfOneImage(image_filename,numPerSz)
        ymins = [ s for s in boxes.ymins() ]
        ymaxs = [ s for s in boxes.ymaxs() ]
        xmins = [ s for s in boxes.xmins() ]
        xmaxs = [ s for s in boxes.xmaxs() ]
        bing_windows = pd.DataFrame({'ymin':ymins,'xmin':xmins,'ymax':ymaxs,'xmax':xmaxs})
        return bing_windows
    def get_iou_distance_matrix(self,bing_windows):
        window_size = bing_windows.shape[0]
        y1 = bing_windows["ymin"].values
        x1 = bing_windows["xmin"].values
        y2 = bing_windows["ymax"].values
        x2 = bing_windows["xmax"].values
        w = x2 - x1 
        h = y2 - y1 
        area = (w*h).astype(float)
        distances = np.zeros((window_size,window_size))
        for i in range(window_size):
            xx1 = np.maximum(x1[i],x1)
            yy1 = np.maximum(y1[i],y1)
            xx2 = np.minimum(x2[i],x2)
            yy2 = np.minimum(y2[i],y2)
            w = np.maximum(0.,xx2-xx1)
            h = np.maximum(0.,yy2-yy1)
            wh = w*h 
            distances[i] = wh/(area[i]+area-wh)
        return distances
    def cluster_boxes(self,bing_windows):
        starttime=time.time()
        distance_matrix = self.get_iou_distance_matrix(bing_windows)
        self.spectral.fit(distance_matrix)
        #get top of each cluster
        window_size = bing_windows.shape[0]
        y1 = bing_windows["ymin"].values
        x1 = bing_windows["xmin"].values
        y2 = bing_windows["ymax"].values
        x2 = bing_windows["xmax"].values
        w = x2 - x1 
        h = y2 - y1
        area=(w*h).astype(float)
        index_dictionary = {}
        for i in range(window_size):
            #if(area[i]<self.min_size):
            #    continue
            #if(w[i]*1.0/h[i]>self.max_ratio or h[i]*1.0/w[i]>self.max_ratio):
            #    continue
            label=self.spectral.labels_[i]
            if not label in index_dictionary:
                index_dictionary[label]=[]
            if len(index_dictionary[label])>=self.top_k:
                continue
            index_dictionary[label].append(i)
        index_list = []
        #for key in index_dictionary:
        #    index_list.extend(index_dictionary[key])
        while True:
            empty=True
            for key in index_dictionary:
                one_list = index_dictionary[key]
                if len(one_list):
                    empty=False
                    index_list.append(one_list.pop(0))
            if empty:
                break
        bing_windows = pd.DataFrame({"ymin":y1[index_list],"xmin":x1[index_list],"ymax":y2[index_list],"xmax":x2[index_list]})
        endtime=time.time()
        #logging.info("Cluster spend {:.3f}".format(endtime-starttime))
        return bing_windows
