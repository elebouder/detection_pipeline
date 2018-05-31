#!ProgramData/Anaconda2/python
import numpy as np
import matplotlib.pyplot as plt
import gdal
from gdal import osr, ogr
import pyproj
import sys
import constants
from PIL import Image
import cv2
import skimage, skimage.io
from multiprocessing import Pool, Process
import Queue
import time
import threading
import collections
import math
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils_ori
import vis_utils
import csv
import tensorflow as tf
import scipy.misc

class Scanner:

    def __init__(self, infile, batchx, batchy):
        print infile
        self.src = gdal.Open(infile)
        self.src_b = self.src.GetRasterBand(1)
        self.src_g = self.src.GetRasterBand(2)
        self.src_r = self.src.GetRasterBand(3)
        self.prj = self.src.GetProjection()
        self.get_proj()
        self.get_keyvals()
        self.batchx = batchx
        self.sizex = (batchx * 25) + 25
        self.batchy = batchy
        self.sizey = (batchy * 25) + 25
        self.window_x = 0
        self.window_y = 0
        self.stop = 0
        print self.sizex
        print self.sizey


    def get_proj(self):
        switcher = {
            'WGS 84 / UTM zone 8N': 32608,
            'WGS 84 / UTM zone 9N': 32609,
            'WGS 84 / UTM zone 10N': 32610,
            'WGS 84 / UTM zone 11N': 32611,
            'WGS 84 / UTM zone 12N': 32612
        }

        srs = osr.SpatialReference(wkt=self.prj)
        if srs.IsProjected():
            projcs = srs.GetAttrValue('projcs')
            print projcs
            self.epsg = switcher.get(projcs, 'nothing')

        if self.epsg == 'nothing':
            print 'projection not in current list of projections handled by this code'
            sys.exit(1)

        # epsg is 8901 for projected, 4326 for decimal lat/lon
        self.pproj = pyproj.Proj(init='epsg:%s' % self.epsg)
        self.praw = pyproj.Proj(init='epsg:4326')

    def get_keyvals(self):
        self.cols = self.src.RasterXSize
        self.rows = self.src.RasterYSize
        print self.cols, self.rows
        self.bands = 3

        geotransform = self.src.GetGeoTransform()
        self.originX = geotransform[0]
        self.originY = geotransform[3]
        self.pixelWidth = geotransform[1]
        self.pixelHeight = geotransform[5]
        self.bandType = gdal.GetDataTypeName(self.src_b.DataType)

    # @profile
    def scan(self):
        xoff = self.window_x
        yoff = self.window_y
        if xoff == -1 or yoff == -1:
            self.arr = None
            return None

        try:
            arr_b = np.array(self.src_b.ReadAsArray(xoff, yoff, self.sizex, self.sizey))
            arr_g = np.array(self.src_g.ReadAsArray(xoff, yoff, self.sizex, self.sizey))
            arr_r = np.array(self.src_r.ReadAsArray(xoff, yoff, self.sizex, self.sizey))

            #self.arr = np.array([arr_b, arr_g, arr_r])
            #self.disp_arr = np.stack([arr_b, arr_g, arr_r], axis=2)
            self.arr = np.stack([arr_b, arr_g, arr_r], axis=2)
        except IndexError as e:
            print'End Of Raster'
            print 'Killing sliding window'
            self.arr = None
            return None

        #self.arr = skimage.img_as_float(self.arr)
        #save_dir = '/home/elebouder/LANDSAT_TF/detection_pipeline/vis/'
        #skimage.io.imsave(save_dir + '{}.png'.format(np.random.random(1)*10), self.arr)
        
        return self.arr

    # @profile
    def next_window(self, count):

        if self.xwindow_canstep():
            self.window_x = self.window_x + (self.sizex - 25)
        elif self.ywindow_canstep():
            self.window_x = 0
            self.window_y = self.window_y + (self.sizey -25)
        else:
            print "Scanner Out of Range"
            self.window_x = -1
            self.window_y = -1

        return self.window_x, self.window_y


    def xwindow_canstep(self):
        distance_left = self.cols - (self.window_x + self.sizex)
        if distance_left < (self.sizex - 25):
            return False
        else:
            return True
 
    def ywindow_canstep(self):
        distance_left = self.rows - (self.window_y + self.sizey)
        if distance_left < (self.sizey - 25):
            return False
        else:
            return True

    """def get_pad_img_coords(self, xoff, yoff, i):
        stepx = (i % self.batchx) * 25 - 25
        stepy = (math.floor(i / self.batchx)) * 25 - 25
        xoff += stepx
        yoff += stepy

        gt = self.src.GetGeoTransform()
        mx = gt[0] + xoff * gt[1]
        my = gt[3] + yoff * gt[5]
        x2, y2 = pyproj.transform(self.pproj, self.praw, mx, my)

        return x2, y2"""

    def get_intrabatch_offsets(self, i):
        
        idx = i % self.batchx
        idy = int(i / self.batchx)
        batch_offsetx = (idx * self.sizex)
        batch_offsety = (idy * self.sizey)
        return batch_offsetx, batch_offsety


    def get_detection_coords(self, xoff, yoff, corners, i):
        xmin, ymin, xmax, ymax = corners
        boffx, boffy = self.get_intrabatch_offsets(i)
        xoffmin = xoff + xmin + boffx
        xoffmax = xoff + xmax + boffx
        yoffmin = yoff + ymin + boffy
        yoffmax = yoff + xmax + boffy
        

        gt = self.src.GetGeoTransform()
        mxmin = gt[0] + xoffmin * gt[1]
        mymin = gt[3] + yoffmin * gt[5]
        mxmax = gt[0] + xoffmax * gt[1]
        mymax = gt[3] + yoffmax * gt[5]

        x21,y21 = pyproj.transform(self.pproj, self.praw, mxmin, mymin)
        x22, y22 = pyproj.transform(self.pproj, self.praw, mxmax, mymax)


        return [x21, y21, x22, y22]



    # @profile
    def getslices(self, datum):
        self.arrlst = []
        iy = 0
        stepx = 25
        stepy = 25
        for n in range(self.batchy):
            ix = 0
            for i in range(self.batchx):
                temp = self.arr[iy:(iy+50), ix:(ix+50), :]
                self.arrlst.append(temp)
                ix += stepx
            iy += stepy

        return self.arrlst


    def close(self):
        self.src = None



class DataTraffic:

    def __init__(self, detectiongraph, sess, scene, scanobj, batch, csv, logfile, categories, category_index, cwd):
        self.cwd = cwd
        self.csv = csv
        self.logfile = logfile
        self.categories = categories
        self.category_index = category_index
        self.detectiongraph = detectiongraph
        self.sess = sess
        self.count = 0
        self.hitlist = []
        self.keepgoing = True
        self.batch = batch
        self.scanobj = scanobj
        self.inputim = scene
        self.idx_win_out = 0
        self.idy_win_out = 0

 
        self.image_tensor = self.detectiongraph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detectiongraph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detectiongraph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detectiongraph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detectiongraph.get_tensor_by_name('num_detections:0')
        self.starttime = time.time()
        self.net_io_control()


    def getstats(self, arr):
        l1 = 0.00
        l2 = 0.00
        for elem in arr:
            l1 += max(elem, -elem)
            l2 += elem**2
        return l1, l2

    # @profile
    def net_io_control(self):
        while self.keepgoing:
            datum = self.scanobj.scan()
            if datum is None:
                self.keepgoing = False
                self.write_coords(self.hitlist)
                print 'done scene'
                break
            #print datum.shape
            if np.amin(datum) == np.amax(datum):
                #save_dir = '/home/elebouder/LANDSAT_TF/detection_pipeline/vis/'
                #skimage.io.imsave(save_dir + '{}.png'.format(np.random.random(1)*10), datum)
                x, y = self.scanobj.next_window(self.count)
                self.idx_win_out = x
                self.idy_win_out = y
                continue
            #dflat = list(datum.flat)
            #dl1, dl2 = self.getstats(dflat)
            #skimage.io.imsave(self.cwd + '/temp.png', datum)
            #nextarr = cv2.imread(self.cwd + '/temp.png')
            #nflat = list(nextarr.flat)
            #nl1, nl2 = self.getstats(nflat)
            arrlist = self.scanobj.getslices(datum)
            #if (dl1 != nl1) or (dl2 != nl2):
            #    print'=========================='
            #    print 'l1: ', dl1, '   ', nl1
            #    print 'l2: ', dl2, '   ', nl2
            #else:
            #    print '+++++++++++++++++'
            #    print 'MATCH'
            #nlist = []
            #for elem in arrlist:
                #save_dir = '/home/elebouder/LANDSAT_TF/detection_pipeline/vis/'
                #skimage.io.imsave(save_dir + '{}.png'.format(np.random.random(1)*10), elem)
                #skimage.io.imsave(self.cwd + '/temp.png', elem)
                #n = cv2.imread(self.cwd + '/temp.png')
                #nlist.append(n)
            #arrlist = nlist
            #TODO remove and account for individual tiles in this arr that are blank, 
            # use np.amax(arr) == np.amin(arr)
            #NOTE with an input arr.len > 1, the tensor has 4 dims and so np.expand_dims not needed
            (boxset, scoreset, classet, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor: arrlist})

            
	    for boxes, classes, scores, arr, i in zip(boxset, classet, scoreset, arrlist, range(len(arrlist))):
                
                detections = ([self.category_index.get(value) for index,value in enumerate(classet[0]) if scoreset[0,index] > 0.8])

                if detections:
                    print detections
                    self.vis_detections(arr, boxes, scores, classes)


                bbox_array, scoremap = vis_utils.get_bbox_coords_on_image_array(arr,
    							np.squeeze(boxes),
    							np.squeeze(classes).astype(np.int32),
    							np.squeeze(scores),
    							self.category_index,
    							use_normalized_coordinates=True,
    							line_thickness=8,
							min_score_thresh=0.80)
	        for elem, score in zip(bbox_array, scoremap):
                    
                    fourcoords = self.scanobj.get_detection_coords(self.idx_win_out, self.idy_win_out, elem, i)
		    #print fourcoords
                    fourcoords.append(score)
                    self.hitlist.append(fourcoords)	
            x, y = self.scanobj.next_window(self.count)
            self.idx_win_out = x
            self.idy_win_out = y


    def calc_centroid(self, elem):
        xmin = elem[0]
        ymin = elem[1]
        xmax = elem[2]
        ymax = elem[3]

        cx = (xmin + xmax)/2
        cy = (ymin + ymax)/2
        return cx, cy



    def write_coords(self, hitlist):
        print len(hitlist)
        print 'writing hitlist'
        elapsed = time.time() - self.starttime
        print elapsed
        with open(self.logfile, 'a') as f:
            f.write("Scene completed:\n" + self.inputim + "\n")
            f.write("# of hits: " + str(len(hitlist)) + " \n")
            f.write("Time elapsed: " + str(elapsed/60) + "\n")
        fieldnames = ['xmin', 'ymin', 'xmax', 'ymax', 'c_x', 'c_y', 'confidence', 'scene']
        with open(self.csv, 'a') as fille:
            writer = csv.DictWriter(fille, fieldnames=fieldnames)
            writer.writeheader()
            for elem in hitlist:
                print elem
                cx, cy = self.calc_centroid(elem)
                xmin = elem[0]
                ymin = elem[1]
                xmax = elem[2]
                ymax = elem[3]
                score = elem[4]
                sid = self.inputim.split("/")[-1]
                writer.writerow({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'c_x': cx, 'c_y': cy, 'confidence': score, 'scene': sid})


    def vis_detections(self, arr, boxes, scores, classes):
        save_dir = '/home/elebouder/LANDSAT_TF/detection_pipeline/vis/'
        #cv2.imwrite(save_dir + '{}.png'.format(np.random.random(1)*10), arr)


        vis_utils_ori.visualize_boxes_and_labels_on_image_array(arr,
                                                        np.squeeze(boxes),
                                                        np.squeeze(classes).astype(np.int32),
                                                        np.squeeze(scores),
                                                        self.category_index,
                                                        use_normalized_coordinates=True,
                                                        line_thickness=3,
                                                        min_score_thresh=0.80)
 
        cv2.imwrite(save_dir + '{}.png'.format(np.random.random(1)*10), arr)
        #plt.figure()
        #plt.imshow(arr)
        #plt.savefig(save_dir + '{}.png'.format(np.random.random(1)*10))

def scan_control(scene, logfile, csv, detectiongraph, sess, categories, category_index, cwd):
    batch = 50
    batchx = 5
    batchy = 10
    #with open(logfile, 'a') as f:
    #    f.write("\nStarting scene\n" + scene + " \n")
    scanobj = Scanner(scene, batchx, batchy)
    DataTraffic(detectiongraph, sess, scene, scanobj, batch, csv, logfile, categories, category_index, cwd)





