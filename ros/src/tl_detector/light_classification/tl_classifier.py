from styx_msgs.msg import TrafficLight
import tensorflow as tf
import scipy
import numpy as np
import cv2
import time
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.graph_def = tf.GraphDef()
        self.graph_def.ParseFromString(tf.gfile.GFile('./frozen_traffic_resnet.pb','rb').read())
        tf.import_graph_def(self.graph_def,name='')
	self.state_name=['Red','Yellow','Green','Unknown']
        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        self.image_tensor = self.graph.get_tensor_by_name('placeholders/inputs:0')
        self.preds = self.graph.get_tensor_by_name('predictions/preds:0')
	self.counter = 0
	self.current_time = time.time()
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	#image_data = scipy.misc.imread(image)
	print('image_shape',image.shape)
	print('image mean',np.mean(image,axis=(0,1)))
        image = cv2.resize(image,(224,224))
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image,axis=0)
        preds = self.sess.run([self.preds],feed_dict={self.image_tensor:image})
        state = preds[0]
        #print(preds)
	self.counter += 1
	#TODO implement light color prediction
        if state == 0:
		state_light = TrafficLight.RED
	elif state == 1: 
		state_light = TrafficLight.YELLOW
	elif state == 2:
		state_light = TrafficLight.GREEN
	else:
		state_light = TrafficLight.UNKNOWN
	#return TrafficLight.UNKNOWN
	if self.counter > 10:
		diff =  time.time() - self.current_time
		#cv2.imwrite('sample_data/'+self.state_name[state]+'_'+str(diff)+'.jpg',np.squeeze(image))
		self.counter = 0
        print('state is ',self.state_name[state])
        #return TrafficLight.RED
	return state_light
