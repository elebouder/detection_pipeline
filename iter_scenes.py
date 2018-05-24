import numpy
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from sliding_win_single import scan_control
import datetime
import tensorflow as tf

class SceneIterator:
    
    def __init__(self, years_dir, cwd, log, data_csv_dir, start_date=None, end_date=None, single_im=False, img_name=None):
        self.years_dir = years_dir
        self.start_date = start_date
        self.end_date = end_date
        self.cwd = cwd
        self.log = log
        self.single_im = single_im
        self.img_name= img_name
        self.data_csv_dir = data_csv_dir
        self.init_ssd_paths()
        self.init_ssd()
        self.control_op()
    
    def get_img_dir(self):
        #get dir for self.img
        #return to control_op
        for root, dirs, files in os.walk(self.years_dir):
            if self.img_name in files:
                return root + '/' +  self.img_name

    def get_nextdir(self, month, year):
        #get next year/month in iteration
        #called when the prev dir exhausted
        #next dir is year or month as appropriate
        #output message to stdout
        #if year, output special message to stdout    
        #save last date to logdir
        #sets next dir
        return os.path.join(self.years_dir, '{}_{}'.format(month, year))
         


    def control_op(self):
        #primary control unit
        #call the next scanner/network control unit
        #outputs to stdout/log
        #passes imdir, logdir, datacsvdir,detection_graph, and sess
        #allow an option to run a single scene with verbose output, or automatically iterate with suppressed output
        if self.single_im:
            im_path = self.get_img_dir()
            month, year = self.date_from_filename()
            csvfile = self.get_next_detectioncsv(month, year)
            print 'starting ', month, ' ', year, ' ', im_path
            scan_control(im_path, self.log, csvfile, self.detection_graph, self.sess, self.categories, self.category_index, self.cwd)
        else:
            dates = self.get_daterange()
            for elem in dates:
                    scene_dir = self.get_nextdir(elem[0], elem[1])
                    csvfile = self.get_next_detectioncsv(elem[0], elem[1])
                    print 'starting ', elem, ' ', scene_dir
                    #TODO write to logdir that we are starting this date
                    for scenefile in os.listdir(scene_dir):
                        if self.already_processed(scenefile):
                            print 'skipping ' + scenefile
                            continue
                        #TODO print starting scene to log
                        print 'Beginning scan of ', scenefile, ' in ', scene_dir
                        scan_control(os.path.join(scene_dir, scenefile), self.log, csvfile, self.detection_graph, self.sess, self.categories, self.category_index, self.cwd)
                        print 'Scan complete.  Moving on...'
                        #TODO print to log that scan is complete


    def already_processed(self, scene_id):
        scenes_completed = []
        with open(self.log, 'r') as f:
            for l in f:
                if ".tif" in l:
                    sid = l.split("/")[-1]
                    scenes_completed.append(sid)


        for elem in scenes_completed:
            idx = scenes_completed.index(elem)
            scenes_completed[idx] = elem.split(" ")[0]
 
        if scene_id in scenes_completed:
            return True
        else:
            return False


    def init_ssd(self):
        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            print 'got here'
            self.sess = tf.Session(graph=self.detection_graph)


    def init_ssd_paths(self):
        self.PATH_TO_CKPT = self.cwd + '/frozen_model/frozen_inference_graph.pb'
        PATH_TO_LABELS = self.cwd + '/lsat_label_map.pbtxt'
        self.NUM_CLASSES = 1
        
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def get_next_detectioncsv(self, month, year):
        return os.path.join(self.data_csv_dir, '{}_{}.csv'.format(month, year))

    def date_from_filename(self):
        assert self.single_im
        im_stripped = self.img_name.split('/')[-1]
        year = im_stripped[9:13]
        julday = im_stripped[13:16]
        month = (datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(julday)-1)).month
        return [month, int(year)]


    def get_daterange(self):
        #given the start and end date, scans the data dirs and returns a list of all [month, year] in that interval (inclusive)
        #FIXME cannot handle dates input in backwards order
        datelist = []
        date1 = self.start_date
        while True:
            if os.path.exists(self.get_nextdir(date1[0], date1[1])):
                datelist.append(date1)
            if date1 == self.end_date:
                break
            else:
                grow_month = self.month_can_grow(date1)
                if grow_month:
                    date1 = [date1[0] + 1, date1[1]]
                else:
                    date1 = [1, date1[1] + 1]
        return datelist


    def month_can_grow(self, date):
        month = date[0]
        if month == 12:
            return False
        else:
            return True
                    
 
def init_pipeline():
    cwd = os.getcwd()
    years_dir = '/media/elebouder/New Volume/Fracking Data/RGB Scenes Ready for Site-Lifting'
    logfile = cwd + '/logs/log.txt'
    data_csv_dir = '/home/elebouder/Data/landsat/detection_csv'
    start_date = [1, 2014]
    end_date = [6, 2014]
    single_im = False
    imgname = 'LC80480192015188LGN00.tif'
    SceneIterator(years_dir, cwd, logfile, data_csv_dir, start_date, end_date, single_im, imgname)


init_pipeline()
