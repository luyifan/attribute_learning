import os
import logging
import time
import optparse
from image_classifier import image_classifier
import scipy.io as sio
PROJECT_DIRNAME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_IMAGE_DATASET = os.path.join(PROJECT_DIRNAME,"JPEGImages")
DEFAULT_CNN_DIRECTORY = os.path.join(PROJECT_DIRNAME,"CNNFEATURE")
logging.getLogger().setLevel(logging.INFO)
class cnn_dataset:
    def __init__(self,dataset_dir=DEFAULT_IMAGE_DATASET,
            cnn_dir=DEFAULT_CNN_DIRECTORY,overwrite=False,
            cluster_num=10,top_k=20):
        logging.info("Initialize CNN BATCH CLASS...")
        self.dataset_dir = dataset_dir
        self.cnn_dir = cnn_dir
        self.overwrite = overwrite
        self.classifier = image_classifier(cluster_num=cluster_num,top_k=top_k)
    def batch_cnn_feature_dataset(self):
        starttime=time.time()
        logging.info("Start batch cnn feature of dataset")
        class_list = os.listdir(self.dataset_dir)
        if not os.path.exists(self.cnn_dir):
            os.mkdir(self.cnn_dir)
        for class_name in class_list:
            class_dir = os.path.join(self.dataset_dir,class_name)
            cnn_of_class_dir = os.path.join(self.cnn_dir,class_name)
            if not os.path.exists(cnn_of_class_dir):
                os.mkdir(cnn_of_class_dir)
            image_list_in_class = os.listdir(class_dir)
            for one_image_file in image_list_in_class:
                image_filename = os.path.join(class_dir,one_image_file)
                prefix = one_image_file.rsplit('.',1)[0]
                cnn_filename = os.path.join(cnn_of_class_dir,prefix+".mat")
                pass_flag=False
                if (not self.overwrite) and (os.path.isfile(cnn_filename)):
                    pass_flag=True
                if pass_flag:
                    #logging.info("{} file exists ,not overwrite".format(cnn_filename))
                    pass
                else:
                    cnn_feature = self.classifier.get_bing_cnn_feature_of_image(image_filename)
                    sio.savemat(cnn_filename, {'cnn_feature':cnn_feature})
                    #logging.info("{} file cnn feature finish".format(cnn_filename))
            logging.info("Finish one class {}".format(class_name))
        logging.info("Finish all dataset spend {:.3f}".format(time.time()-starttime))
            
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-d','--dir',
            help="the image dataset directory",
            metavar="FILE", default=DEFAULT_IMAGE_DATASET)
    parser.add_option('-s','--store',
            help="the directory to store cnn feature",
            metavar="FILE", default=DEFAULT_CNN_DIRECTORY)
    parser.add_option('-o','--overwrite',
            help="enable overwrite cnn file, default False",
            action="store_true",default=False)
    parser.add_option('-c','--cluster_num',
            help="the cluster num for bing normalized-cut",
            type='int', default=10)
    parser.add_option('-k','--top_k',
            help="the top k bing score for each cluster",
            type='int', default=20)
    opts,args = parser.parse_args()
    batch_operation = cnn_dataset(dataset_dir=opts.dir,cnn_dir=opts.store,
            overwrite=opts.overwrite,cluster_num=opts.cluster_num,top_k=opts.top_k)
    batch_operation.batch_cnn_feature_dataset()
