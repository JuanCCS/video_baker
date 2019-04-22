import cv2
import numpy as np
import time
import os
import sys
from math import sqrt
import pandas as pd 
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import pickle
import matplotlib.ticker as tkr 

directory = os.path.dirname(__file__)
roi_df = pickle.load(open('/Users/juancachafeiro/Desktop/FitWave/VideoBaker/BaseVideo/melfi40Yd.p', 'rb'))

'''
	The Run Code
'''

def numfmt(x, pos): # your custom formatter function: divide by 100.0
    s = '{:.0f}'.format(x / 60)
    return s

    # has classes for tick-locating and -formatting
yfmt = tkr.FuncFormatter(numfmt)    # create your custom formatter function

# your existing code can be inserted here



# Assumes a camera FOV of 94.4deg, r and c distances are in yards
def camera_to_meters(camera_x, global_data):
	run_length = global_data.run_zone_end - global_data.run_zone_start
	yards_x = ((camera_x - global_data.run_zone_start)/run_length)*40
	cartesian_x = yards_x - 20
	if cartesian_x == 0:
		return 20
	#radius of circle
	r = 27.258
	# distance from camera to center of run
	c = 8.74
	projected_x = (r*cartesian_x)/(c+sqrt(r**2 - cartesian_x**2))
	meters_x = 0.9144*(projected_x+20)
	return meters_x

class Run:
	start_frame = 0
	end_frame = 0
	centers=[]
	frames=[]
	run_id = ''

	def __init__(self, inputs):
		self.centers=[]
		self.frames=[]
		self.start_frame = inputs.frame_count
		self.centers.append(inputs.current_center[0])
		self.frames.append(inputs.frame_count)
		self.run_id = len(inputs.past_runs)

	def close(self, inputs):
		self.end_frame = inputs.frame_count

	def build_prediction(self):
		return 0

	def build_graphs_until_frame(self, frame):
		return 0

	def log_data(self):
		for x in zip(self.centers, self.frames):
			print('Run {} Center:{} Run Frame:{}'.format(self.run_id, x[0], x[1]))
		print('Elapsed_Time: {}'.format((self.end_frame-self.start_frame)/60))

	def render(self, inputs):
		x = self.frames
		y = self.centers
		x = list(map(lambda i, s = rn.start_frame: i - s + 1, x))
		run_duration = rn.end_frame - rn.start_frame

		z = np.polyfit(x, y, 4)
		p = np.poly1d(z)
		v = np.poly1d.deriv(p)

		xp = np.linspace(1, run_duration, run_duration)
		plt.plot(xp, p(xp), '-')
		plt.plot(xp, v(xp), '.')
		plt.ylim(ymin=0,ymax=2000)
		plt.show()


class Globals:
	current_video = ''
	past_runs = []
	current_run = ''
	frame_count = 0
	current_center = 0
	run_zone_start = 0
	run_zone_end = 0
	input_x = 3840
	input_y = 2160
	test_x = 720
	test_y = 576
	output_x = 1080
	output_y = 1350	
	video_location = ''
	track_width_pixels = 0
	track_width_meters = 0
	pixels_per_meter= 0

'''
	The Video Baker Code
'''

class State:
	def run(self):
		assert 0, "run not implemented"
	def transition(self, input):
		assert 0, "next not implemented"


class StateMachine:
	def __init__(self, initialState):
		self.currentState = initialState
		self.currentState.run()
	# Template method:
	def runAll(self, inputs):
		self.currentState = self.currentState.transition(inputs)
		self.currentState.run()

class Waiting(State):
	run_count = 0
	#Runs until a runner's center is past the run_zone_start position.
	def run(self):
		#print('Waiting For Run')
		return 0

	def transition(self, input):
		if input.current_center[0] > input.run_zone_start:
			if input.current_center[0] > input.run_zone_end:
				return VideoBaker.waiting
			self.run_count += 1
			#print('run found')
			found_run = Run(global_data)
			global_data.current_run = found_run
			return VideoBaker.measuring
		return VideoBaker.waiting

class Measuring(State):
	#Runs until a runner's center is past the run_zone_end position.
	def run(self):
		global_data.current_run.centers.append(global_data.current_center[0])
		global_data.current_run.frames.append(global_data.frame_count)

	def transition(self, input):
		if input.current_center[0] > (input.run_zone_end - 80):
			global_data.current_run.close(global_data)
			global_data.past_runs.append(global_data.current_run)
			#print('run_logged')
			return VideoBaker.waiting
		return VideoBaker.measuring

# class Baking(State):
# 	# Bakes the measurements into a new video with all the data incorporated in the video
# 	# Max_Speed, Max_Acceleration, Graphs, All Mounted on the Video
# 	def run(self):
# 		bake_frame(frame)

# 	def transition(self, input):
# 		if input == MouseAction.appears:
# 			return MouseTrap.luring
# 		return MouseTrap.waiting


class VideoBaker(StateMachine):
    def __init__(self):
        # Initial state
        StateMachine.__init__(self, VideoBaker.waiting)

VideoBaker.waiting = Waiting()
VideoBaker.measuring = Measuring()
video_baker = VideoBaker()
global_data = Globals()

# Select The run Zone and Assign it to The Globals Variable
global_data.video_location = '/Users/juancachafeiro/Desktop/FitWave/VideoBaker/BaseVideo/melfi40Yd.mp4'
current_video = cv2.VideoCapture(global_data.video_location)
ret,frame = current_video.read()
if ret == False:
	sys.exit('Failed to read video on ROI Selection Phase')

resized_frame = ''
scale_factor = 0.25
resized_frame = cv2.resize(frame, dsize=(int(global_data.input_x*scale_factor), int(global_data.input_y*scale_factor)))

showCrosshair = False
fromCenter = False
r = cv2.selectROI("Image", resized_frame, fromCenter, showCrosshair)
track_width_resized = r[2]
global_data.track_width_pixels = track_width_resized*(1/scale_factor)
global_data.track_width_meters = 36.576
global_data.pixels_per_meter = global_data.track_width_pixels / global_data.track_width_meters

# Mark begin and End of Track
global_data.run_zone_start = r[0] * (1/scale_factor)
#print(global_data.run_zone_start)
global_data.run_zone_end = (r[0] + r[2]) * (1/scale_factor)
#print(global_data.run_zone_end)
current_video.release()

for frame in roi_df[0]:
	
	roi_array = frame['rois']
	global_data.frame_count += 1
	for roi in roi_array:
		#print(roi)
		# Adjust Rois for Video Size Output. Using Test Measurements
		y1 = int((roi[0]/global_data.test_x)*global_data.input_y)
		y2 = int((roi[2]/global_data.test_x)*global_data.input_y)
		x1 = int(((roi[1])/global_data.test_y)*global_data.input_x)
		x2 = int(((roi[3])/global_data.test_y)*global_data.input_x)
		print('{}, {}'.format(x1, x2))
		w = np.absolute(x2-x1)
		h = np.absolute(y2-y1)
		tan = w/h
		atan = h/w

		#if y1 < (global_data.input_y * 0.4):
		#	continue

		x1+=int((x1-x2)//2)
		y1+=int((y2-y1)//2)
		global_data.current_center = (x1,y1)
		#print(global_data.current_center)

		video_baker.runAll(global_data)

count = 0
for rn in global_data.past_runs:
	x = rn.frames
	y = rn.centers
	rn.end_frame += 0
	x = list(map(lambda i, s = rn.start_frame: i - s + 1, x))
	run_duration = rn.end_frame - rn.start_frame
	projected_y = list(map(lambda i: camera_to_meters(i, global_data), y))
	
	w = np.polyfit(x, projected_y, 9)
	z = np.polyfit(x, y, 9)
	pixel_predictor = np.poly1d(z)
	meters_predictor = np.poly1d(w)
	velocity_predictor = np.poly1d.deriv(meters_predictor)
	accel_predictor = np.poly1d.deriv(velocity_predictor)
	xp = np.linspace(1, run_duration, run_duration)

	current_video = cv2.VideoCapture(global_data.video_location)
	fourcc = cv2.VideoWriter_fourcc(*'a\0\0\0')
	output_vid_file_name = '/Users/juancachafeiro/Movies/AdidasCV/melfi40Yd{}.mp4'.format(rn.run_id)
	if os.path.isfile(output_vid_file_name):
		os.remove(output_vid_file_name)
	out = cv2.VideoWriter(output_vid_file_name,fourcc, 15, (global_data.output_x, global_data.output_y))
	font_path = os.path.join(directory, 'AdiHaus/AdiHausDIN-Bold.ttf')
	metrics_font = ImageFont.truetype(font_path, 70)
	metrics_fill = (255,250,22,255)

	count = 0
	sub_count = 0
	speed_array = []
	mean_speed = 0
	prev_x = meters_predictor(0)
	prev_speed = 0
	max_speed = 0
	mean_speed_agg = []
	while True:
		count += 1
		ret,frame = current_video.read()
		if ret == False:
			print('Video broke while rendering')
			break
		if count > rn.start_frame:
			if count < rn.end_frame:
				
				#Calculate current center Using Approximation
				current_length = count-rn.start_frame + 1
				time = (current_length) / 60
				vid_x = int(np.floor(pixel_predictor(current_length)))
				meters_x = meters_predictor(current_length)
				center=(vid_x, int(global_data.output_y*0.5))
				cut_y = center[1]-(global_data.output_y//2)
				cut_x = center[0]-(global_data.output_x//2)

				if cut_y < 1:
					cut_y = 1
					#print('Cut Y: {}'.format(cut_y))
				if cut_y + global_data.output_y > global_data.input_y:
					cut_y = global_data.input_y - global_data.output_y - 1
					#print('Cut Y: {}'.format(cut_y))
				if cut_x < 1:
					cut_x = 1
					#print('Cut X: {}'.format(cut_x))
				if cut_x + global_data.output_x > global_data.input_x:
					cut_x = global_data.input_x - global_data.output_x - 1
					#print('Cut X: {}'.format(cut_x))

				cropped_frame = frame[cut_y : cut_y+global_data.output_y, cut_x : cut_x + global_data.output_x]
				pil_img = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
				d = ImageDraw.Draw(pil_img)

				#Calculate speed
				distance = meters_x - prev_x
				meters_speed = distance/(1/60)
				kmh_speed = meters_speed * 3.6 #m/s to km/h
				accel = meters_speed - prev_speed
				prev_speed = kmh_speed
				prev_x = meters_x

				sub_count += 1 
				speed_array.append(kmh_speed)
				if sub_count > 3:
					sub_count = 0
					mean_speed = np.mean(speed_array)
					mean_speed_agg.append(mean_speed)
					speed_array = []

				if mean_speed > max_speed:
					if meters_x > 27:
						mean_speed = max_speed
					else:
						max_speed = mean_speed
				graph_x = np.linspace(1, current_length, current_length)
				fig = plt.figure(frameon=False)
				ax = fig.add_axes([0, 0, 1, 1])
				ax.set_ylim(ymin=0, ymax=40)
				ax.set_xlim(xmin=0, xmax=run_duration)
				graphcolor = '#FFFA16'
				ax.set_xlabel('Tiempo (s)')
				ax.set_ylabel('Distancia (m)')
				all_spines = ['left', 'right', 'top', 'bottom']
				for spine in all_spines:
					ax.spines[spine].set_color(graphcolor)
					ax.spines[spine].set_linewidth(4)
				for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
					item.set_fontsize(20)
				ax.xaxis.label.set_color(graphcolor)
				ax.xaxis.set_major_formatter(yfmt)
				ax.yaxis.label.set_color(graphcolor)
				ax.tick_params(axis='x', colors=graphcolor)
				ax.tick_params(axis='y', colors=graphcolor)
				ax.plot(graph_x, meters_predictor(graph_x), '-', linewidth=8, color=graphcolor)
				fig.savefig('demo.png', transparent=True, bbox_inches='tight')
				plt.close()
				dmo = Image.open('demo.png')
				ratio = 0.675
				halfsize = tuple(int(s*ratio) for s in dmo.size)
				dmo.thumbnail(halfsize, Image.ANTIALIAS)
				top_corner_x = int(global_data.output_x - dmo.size[0] - 20)
				top_corner_y = int(global_data.output_y - dmo.size[1] - 20)
				pil_img.paste(dmo, (top_corner_x, top_corner_y), mask=dmo)

			
				d.text((20, 400), 'Velocidad {:.0f} Km/h'.format(mean_speed), font=metrics_font, fill=metrics_fill)
				d.text((20, 475), 'Tiempo {:.2f} s'.format(time), font=metrics_font, fill=metrics_fill)
				out.write(cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR))
			else:
				print(np.mean(mean_speed_agg))
				break

	current_video.release()
	out.release()