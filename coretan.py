
import numpy as np
from urllib.request import urlopen
import time
import cv2

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

time_st = time.time()
url = 'https://upload.wikimedia.org/wikipedia/commons/0/06/Common_zebra.jpg'
print(url_to_image(url))
print(time.time() - time_st)