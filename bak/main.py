import cv2
import pysift
import numpy as np
from numpy import float32
def warpTwoImages(img2, img1, H):
	'''warp img2 to img1 with homograph H'''
	print("=={}==".format(H))
	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	pts1 = float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	print(pts1)
	pts2 = float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	print(pts2_)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
	result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
	result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
	return result

def DoMerge(imgA, imgB, H):
	#a_lu,a_ru,a_ld,a_rd;
	a_x,a_y = imgA.shape
	b_x,b_y = imgB.shape
	shapeA = np.matrix([[0,0,1], [0,a_y,1],[a_x,0,1],[a_x,a_y,1]]).T
	shapeB = np.matrix([[0,0,1], [0,b_y,1],[b_x,0,1],[b_x,b_y,1]]).T
	H = np.matrix(H)
	#print(H)
	#print(shapeA)
	shapeA = H.dot(shapeA)
	print(shapeA.shape)
	for i in range(4):
		print(shapeA[2])
		exit(1)
		shapeA[0][0][i] /= shapeA[2][0][i]	
		shapeA[1][0][i] /= shapeA[2][0][i]	
		shapeA[2][0][i] = 1	
	#print(shapeA)
	#print(shapeB)
	
	#b_lu,b_ru,b_ld,b_rd;
	
def MergeImg(leftImg, rightImg):
	left = cv2.imread(leftImg, 0)
	right = cv2.imread(rightImg, 0)
	
	rate = 0.05

	# resize src img
	l, w = left.shape
	print('first image shape [{},{}]'.format(l,w))
	left = cv2.resize(left, (int(w*rate), int(l*rate)))
	l, w = left.shape
	print('resize to [{},{}]'.format(l, w))
	
	l, w = right.shape
	print('second image shape [{},{}]'.format(l,w))
	right = cv2.resize(right, (int(w*rate), int(l*rate)))
	l, w = right.shape
	print('resize to [{},{}]'.format(l, w))

	#cv2.imshow('left', left)
	#cv2.waitKey(0)

	#cv2.imshow('right', right)
	#cv2.waitKey(0)

	if left is None or right is None:
		print("open src img failed")
		exit(1)
	# detactive key opints
	print('finding key points of first image...')
	left_kps, left_sifts = pysift.computeKeypointsAndDescriptors(left)
	print('finding key points of second image...')
	right_kps, right_sifts = pysift.computeKeypointsAndDescriptors(right)

	#match kps
	print('matching key points...')
	matcher = cv2.DescriptorMatcher_create('BruteForce')
	rawMatches = matcher.knnMatch(left_sifts, right_sifts, 2)

	matches = []
	for m in rawMatches:
		#print('....')
		if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
			matches.append((m[0].trainIdx, m[0].queryIdx))
			
	#print(matches)
	#calculate matrix H which let   Hx = x_
	
	# construct x matrix
	matrix_size = 4
	'''
	left_x = []
	right_x = []
	for i in range(matrix_size):
		index = matches[i][0]
		tmp = []
		tmp.append(left_kps[index].pt[0])
		tmp.append(left_kps[index].pt[1])
		tmp.append(1)
		left_x.append(tmp)
		
		index = matches[i][1]
		tmp = []
		tmp.append(right_kps[index].pt[0])
		tmp.append(right_kps[index].pt[1])
		tmp.append(1)
		right_x.append(tmp)

	left_x = np.matrix(left_x).T
	right_x = np.matrix(right_x).T
	
	H = right_x.dot(left_x.I)
	print(H)
	'''
	ptsA = np.float32([left_kps[i].pt for (_, i) in matches])		
	ptsB = np.float32([right_kps[i].pt for (i, _) in matches])		
	H, statuc = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
	result = warpTwoImages(left, right, H)
	#DoMerge(left, right, H)
	#result = cv2.warpPerspective(left, H, (left.shape[1] + right.shape[1], left.shape[0]))
	#result[0:right.shape[0], right.shape[1]:right.shape[1]*2] = right
	cv2.imshow('result', result)
	cv2.waitKey(0)
if __name__ == "__main__":
	MergeImg('left.jpg', 'right.jpg')