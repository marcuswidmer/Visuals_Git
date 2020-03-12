import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def normalize(image):
	image = image - image.min()
	return (image/image.max() * 255).astype(np.uint8)


filename = 'lwjbfrofwd.mov'

of = os.path.splitext(filename)[0]
fex = os.path.splitext(filename)[1]

vidcap = cv2.VideoCapture(filename)
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*'H','2','6','4')
out = cv2.VideoWriter(of + '_gradient.mp4',fourcc, fps, (width, height))

i = 0
while (vidcap.isOpened()):
	ret, frame = vidcap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	[Ax, Ay] = np.gradient(frame)
	norm_img = normalize(Ax)
	tiled = np.tile(norm_img.reshape(height, width,1),[1,1,3])
	out.write(tiled)
	i += 1
	if np.mod(i,100) == 0:
		plt.clf()
		plt.imshow(tiled)
		plt.pause(0.001)

	print('Rendering gradientvideo %s%%' %(int(i/(length-1)*100)), end='\r')
	# plt.clf()
	# plt.imshow(np.tile(norm_img.reshape(height, width,1),[1,1,3]))
	# plt.pause(0.001)




