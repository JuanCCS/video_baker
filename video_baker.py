import cv2
import numpy as np
import time
import os
import pandas as pd 
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

directory = os.path.dirname(__file__)
roi_df = pd.read_pickle('/Users/juancachafeiro/Desktop/VideoBaker/my_df (1).p')
print(roi_df.head())

video = cv2.VideoCapture('/Users/juancachafeiro/Movies/Us_CV/Unprocessed_Video.mp4')

centers = pd.DataFrame(columns=['Frame', 'Center'])


def pixels_to_meters(pixel_value, pixels_per_meter):
	return pixel_value/pixels_per_meter

vid_x = 720
vid_y = 480

output_x = 1920
output_y = 1080

count = 0
start_x = 0
start_time = 0
end_time = 0
prev_x1 = 0
for index, row in roi_df.iterrows():
	ret,frame = video.read()
	if ret == False:
		break;
	roi_array = row['Roi']

	for roi in roi_array:
		# Adjust Rois for Video Size Output
		y1 = int((roi[0] / vid_x)*output_y)
		y2 = int((roi[2] / vid_x)*output_y)
		x1 = int((roi[1] / vid_y)*output_x)
		x2 = int((roi[3] / vid_y)*output_x)
		w = np.absolute(x2-x1)
		h = np.absolute(y2-y1)
		tan = w/h
		atan = h/w
		#First, check for rois that are outside of the view bounds
		if y1 < output_y * 0.4:
			continue
		# if tan > 3 or atan > 3:
		# 	continue
		# if prev_x1 != 0 and np.absolute(x1-prev_x1) > output_x * 0.1:
		# 	continue
		#print('Frame: {}, X1: {}, X2: {}, Y1: {}, Y2:{}'.format(count,x1,x2,y1,y2))
		cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
		prev_x1 = x1
		x1+=int((x2-x1)//2)
		y1+=int((y2-y1)//2)
		center = (x1,y1)
		cv2.circle(frame,center, 3, (0,0,255), -1)
		centers = centers.append({'Frame': count, 'Center': center[0]}, ignore_index=True)
		if start_x == 0:
			start_x = center[0]
			start_time = count
			
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	count+=1


video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

x = centers['Frame'].tolist()
y = centers['Center'].tolist()

z = np.polyfit(x, y, 8)
p = np.poly1d(z)

xp = np.linspace(1, video_length, video_length)


plt.scatter(x, y, s=200, alpha = 0.8)
plt.ylim(ymin=0,ymax=2000)
plt.plot(xp, p(xp), '-')
plt.show()

#Select Track
video.release();
video = cv2.VideoCapture('/Users/juancachafeiro/Movies/Us_CV/Unprocessed_Video.mp4')

ret,frame = video.read()

showCrosshair = False
fromCenter = False
r = cv2.selectROI("Image", frame, fromCenter, showCrosshair)
track_width = r[2]

video.release();
video = cv2.VideoCapture('/Users/juancachafeiro/Movies/Us_CV/Unprocessed_Video.mp4')
track_half_width = track_width//2

center_of_track = start_x
distance_of_track_in_meters = 10
pixels_per_meter = track_width/distance_of_track_in_meters

# BRAND THE VIDEO
count = 1
sub_count = 0
prev_x = start_x
prev_speed = 0

baked_video_x = 576
baked_video_y = 720

fourcc = cv2.VideoWriter_fourcc(*'a\0\0\0')
out = cv2.VideoWriter('/Users/juancachafeiro/Movies/Us_CV/output.mp4',fourcc, 15, (baked_video_x,baked_video_y))
font_path = os.path.join(directory, '/Roboto_Condensed/RobotoCondensed-Bold.ttf')
metrics_font = ImageFont.truetype(font_path, 50)
fill_name = (249,29,67,255)
final_speed=0
sub_count = 0
while True:
	ret,frame = video.read()
	if ret == False:
		break;
	vid_x = int(np.floor(p(count)))
	if vid_x > output_x: 
		vid_x = start_x
	if vid_x < 0:
		vid_x = start_x
	#calc_speed
	distance = vid_x - prev_x
	pixel_speed = pixels_to_meters((distance/(1/60)), pixels_per_meter)*3.6
	sped_diff = pixels_to_meters((pixel_speed - prev_speed), pixels_per_meter)
	
	center=(vid_x, output_y//2)
	print(vid_x)
	y = center[1]-(baked_video_y//2)
	print('CropY: {}'.format(y))
	x = center[0]-(baked_video_x//2)
	print('CropX: {}'.format(x))
	count+=1
	if y < 0 or y + baked_video_y > output_y:
		continue
	if x < 0 or x + baked_video_x > output_x:
		continue

	cropped_frame = frame[y:y+baked_video_y, x:x+baked_video_x]
	#cv2.circle(frame,center, 63, (0,0,255), -1)
	#cv2.putText(cropped_frame,'Speed: {:.0f} km/h'.format(pixel_speed),(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1,cv2.LINE_AA)
	pil_img = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
	d = ImageDraw.Draw(pil_img)
	sub_count+=1
	if sub_count > 5:
		sub_count = 0
		final_speed = int(pixel_speed)
	t = (count-start_time)/60
	if t < 0:
		t = 0
	d.text((20, 100), '{0:.0f} Km/h'.format(final_speed), font=metrics_font, fill=fill_name)
	d.text((20, 175), '{:.2f} s'.format(t), font=metrics_font, fill=fill_name)
	out.write(cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR))
	prev_x = vid_x
	prev_s = pixel_speed
	del d
	if cv2.waitKey(4) & 0xFF == ord('q'):
		break

video.release()
out.release()
