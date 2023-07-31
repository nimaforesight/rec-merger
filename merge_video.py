from pickletools import uint8
from re import L
import sqlite3
import cv2
import os
import sys
from moviepy.editor import * #VideoFileClip, clips_array, vfx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from numpy.core.fromnumeric import clip
from numpy.core.records import array
from scipy.interpolate import make_interp_spline
import logging
import datetime
import time
import ffmpeg
from timezonefinder import TimezoneFinder
from dateutil import tz
import pytz
from dotenv import load_dotenv
from PIL import ImageFont, ImageDraw, Image

class video_merger:
    def __init__(self, video):
        load_dotenv('/env/.env')
        self.video_name = video
        self.db = {'ml': [], 'recording': [], 'gps': []}
        self.tmp_dir = 'tmp/'
        self.icons_dir = 'icons/'
        pedal_icon_dim = (42, 44)
        indicator_icon_dim = (46, 38)
        self.icons = {
            'wheel': cv2.imread(self.icons_dir+'wheel_v2.jpg'),
            'red_pedal': cv2.resize(cv2.imread(self.icons_dir+'red-pedal.jpg'), pedal_icon_dim),
            'green_pedal': cv2.resize(cv2.imread(self.icons_dir+'green-pedal.jpg'), pedal_icon_dim),
            'foresight_logo': cv2.imread(self.icons_dir+'logo.jpg'),
            'interior_covered': cv2.imread(self.icons_dir+'camera_covered.jpg'),
            'speed': cv2.imread(self.icons_dir+'speed_v2.jpg'),
            'right_indicator_on': cv2.resize(cv2.imread(self.icons_dir+'right_indicator_on.png'), indicator_icon_dim),
            'left_indicator_on': cv2.resize(cv2.imread(self.icons_dir+'left_indicator_on.png'), indicator_icon_dim),
            'right_indicator_off': cv2.resize(cv2.imread(self.icons_dir+'right_indicator_off.png'), indicator_icon_dim),
            'left_indicator_off': cv2.resize(cv2.imread(self.icons_dir+'left_indicator_off.png'), indicator_icon_dim)
        }
        self.wheel_font = ImageFont.truetype("./fonts/Cutive-Regular.ttf", 16)
        self.speed_font = ImageFont.truetype("./fonts/Cutive-Regular.ttf", 24)
        self.speed_unit_font = ImageFont.truetype("./fonts/Cutive-Regular.ttf", 10)
        self.traffic_light_circle_size = 10
        self.green_light = (23,239,16)
        self.red_light = (0,0,255)
        self.date_time_color = (26,26,203)
        self.recording_name_color = (26,26,203)
        self.date_time_font_size = 0.6
        self.recording_name_font_size = 0.6
        self.mlstreamer_versioon = '0.15.0'
        self.recording_version = '0.7.2'
        self.aws_id = os.environ.get('AWS_ID')
        self.aws_upload_path = os.environ.get('REC_MERGES_DIR')
        self.nas_dir = os.environ.get('LOCAL_MERGED_DIR')
        self.logger = logging.getLogger()
        self.recording_name = None
        self.recording_path = None
        self.red = (71, 80, 183)
        self.green = (175, 230, 126)
        self.messages = []

    def get_db_data(self, db_file):
        conn = None
        conn = sqlite3.connect(self.recording_path + db_file + '.db')
        self.db[db_file] = conn.cursor()

    def get_frames(self):
        cur = self.db['ml']
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='frame';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        if table_exist == 1:
            columns = [i[1] for i in cur.execute('PRAGMA table_info(frame)')]
            if columns[0] != 'camera_id':
                frames = False
            else:
                cur.execute("SELECT timestamp,frame_num FROM frame WHERE camera_id=?", (1,))
                frames = cur.fetchall()
        else:
            frames = False
            if frames != False and len(frames) == 0:
                frames = False
        return frames

    def get_wheel(self, frames):
        cur = self.db['recording']
        cur.execute("SELECT timestamp FROM ccan_wheel")
        timestamps = cur.fetchall()
        if len(timestamps) <= 0:
            return []
        times = list()
        for i in range(len(timestamps)):
            times.append(timestamps[i][0])
        frameList = list()
        for i in range(len(frames)):
            frameList.append(frames[i][0])

        wheel = list()
        for i in range(len(frames)):
            r = self.closest(times,frameList[i])
            query = "SELECT CAST(rotation AS INT) FROM ccan_wheel where timestamp="+str(r)
            cur.execute(query)
            wheelData = cur.fetchall()
            wheel.append(wheelData[0][0])
        return wheel

    def get_speed(self):
        cur = self.db['ml']
        cur.execute("SELECT timestamp FROM frame where camera_id=?",(1,))
        timestamp = cur.fetchall()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        if table_exist == 1:
            cur.execute("SELECT speed FROM vehicle")
            speedData = cur.fetchall()
            speed = list()
            if speedData[0][0] == None or len(timestamp) != len(speedData):
                speed = self.get_speed_recording(timestamp)
            else:
                speed = self.handle_null_values(speedData)
        else:
            speed = self.get_speed_recording(timestamp)
        return speed

    def get_speed_recording(self, timestamp):
        cur = self.db['recording']
        cur.execute("SELECT timestamp FROM ccan_speed")
        timestamp_speed = cur.fetchall()
        speed = list()
        if len(timestamp_speed) > 0:
            for time in timestamp:
                res = self.find_nearest(timestamp_speed,time)[0]
                query = "SELECT speed FROM ccan_speed where timestamp="+str(res)
                cur.execute(query)
                speed.append(round(cur.fetchone()[0]))
        else:
            for time in timestamp:
                speed.append(0)
        return speed

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def get_brake(self):
        cur = self.db['ml']
        cur.execute("SELECT timestamp FROM frame where camera_id=?",(1,))
        timestamp = cur.fetchall()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        if table_exist == 1:
            cur.execute("SELECT brake FROM vehicle")
            brakeData = cur.fetchall()
            brake = list()
            if brakeData[0][0] == None or len(timestamp) != len(brakeData):
                brake = self.get_brake_recording(timestamp)
            else:
                brake = self.handle_null_values(brakeData)
        else:
            brake = self.get_brake_recording(timestamp)
        return brake

    def get_brake_recording(self, timestamp):
        cur = self.db['recording']
        cur.execute("SELECT timestamp FROM ccan_brake")
        timestamp_gas = cur.fetchall()
        brake = list()
        if len(timestamp_gas) > 0:
            for time in timestamp:
                res = self.find_nearest(timestamp_gas,time)[0]
                query = "SELECT depression FROM ccan_brake where timestamp="+str(res)
                cur.execute(query)
                brake.append(round(cur.fetchone()[0]))
        else:
            for time in timestamp:
                brake.append(0)
        return brake

    def get_gas(self):
        cur = self.db['ml']
        cur.execute("SELECT timestamp FROM frame where camera_id=?",(1,))
        timestamp = cur.fetchall()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        if table_exist == 1:
            cur.execute("SELECT gas FROM vehicle")
            gasData = cur.fetchall()
            gas = list()
            if gasData[0][0] == None or len(timestamp) != len(gasData):
                gas = self.get_gas_recording(timestamp)
            else:
                gas = self.handle_null_values(gasData)
        else:
            gas = self.get_gas_recording(timestamp)
        return gas

    def handle_null_values(self, values):
        res = []
        for i in range(len(values)):
            if values[i][0] == None:
                res.append(correct_value)
            else:
                correct_value = round(values[i][0])
                res.append(correct_value)
        return res
    
    def get_gas_recording(self, timestamp):
        cur = self.db['recording']
        cur.execute("SELECT timestamp FROM ccan_gas")
        timestamp_gas = cur.fetchall()
        gas = list()
        if len(timestamp_gas) > 0:
            for time in timestamp:
                res = self.find_nearest(timestamp_gas,time)[0]
                query = "SELECT depression FROM ccan_gas where timestamp="+str(res)
                cur.execute(query)
                gas.append(round(cur.fetchone()[0]))
        else:
            for time in timestamp:
                gas.append(0)
        return gas

    def get_left_indicator(self):
        cur = self.db['ml']
        cur.execute("SELECT timestamp FROM frame where camera_id=?",(1,))
        timestamp = cur.fetchall()
        totalFrames = len(timestamp)
        indicator = list()
        cur = self.db['recording']
        query = "SELECT timestamp,active FROM ccan_left_indicator"
        cur.execute(query)
        indicator.append(cur.fetchall())
        res_indicator = [0]*totalFrames
        if len(indicator[0]) > 1:
            for item in range(len(indicator[0])):
                j = 0
                for time in timestamp:
                    if time[0] > indicator[0][item][0]:
                        res_indicator[j] = indicator[0][item][1]
                    j = j +1
        elif len(indicator[0]) == 1 and indicator[0][0][1] == 1:
            for item in range(len(indicator[0])):
                j = 0
                for time in timestamp:
                    if time[0] > indicator[0][item][0]:
                        res_indicator[j] = indicator[0][item][1]
                    j = j +1
        return res_indicator

    def get_right_indicator(self):
        cur = self.db['ml']
        cur.execute("SELECT timestamp FROM frame where camera_id=?",(1,))
        timestamp = cur.fetchall()
        totalFrames = len(timestamp)
        indicator = list()
        cur = self.db['recording']
        query = "SELECT timestamp,active FROM ccan_right_indicator"
        cur.execute(query)
        indicator.append(cur.fetchall())
        res_indicator = [0]*totalFrames
        if len(indicator[0]) > 1:
            for item in range(len(indicator[0])):
                j = 0
                for time in timestamp:
                    if time[0] > indicator[0][item][0]:
                        res_indicator[j] = indicator[0][item][1]
                    j = j +1
        
        elif len(indicator[0]) == 1 and indicator[0][0][1] == 1:
            for item in range(len(indicator[0])):
                j = 0
                for time in timestamp:
                    if time[0] > indicator[0][item][0]:
                        res_indicator[j] = indicator[0][item][1]
                    j = j +1
        return res_indicator

    def create_pedal_bar(self, value, color, margin=3, background=167, height=73, width=43):
        margin = max(0, min(width//2, margin))
        # Clamp gas value bewteen [0, 99]
        percent = max(0, min(100, value)) / 100
        value = int(percent * (height - margin*2))
        # Create a grey background
        image = np.full((height,width,3), background, dtype=np.uint8)
        # Fill in `color` by `gas` amount
        image[height-margin-value:height-margin, margin:width-margin] = color 
        return image

    def get_late_brakes(self):
        cur = self.db['ml']
        cur.execute("SELECT late_brake FROM brake_model")
        lateBrakeData = cur.fetchall()
        lateBrake = list()
        for i in range(len(lateBrakeData)):
            lateBrake.append(lateBrakeData[i][0])
        return lateBrake

    def get_late_start(self):
        cur = self.db['ml']
        cur.execute("SELECT late_start FROM brake_model")
        lateStartData = cur.fetchall()
        lateStart = list()
        for i in range(len(lateStartData)):
            lateStart.append(lateStartData[i][0])
        return lateStart

    def get_follow_too_close(self):
        cur = self.db['ml']
        cur.execute("SELECT follow_too_close FROM brake_model")
        followData = cur.fetchall()
        follow = list()
        for i in range(len(followData)):
            follow.append(followData[i][0])
        return follow

    def find_traffic_light(self, frame):
        cur = self.db['ml']
        cur.execute("SELECT final_output,x,y,width,height FROM light_model where frame_num=?",(frame,))
        light = cur.fetchall()
        return light

    def rotation(self, img, angle):
        if angle > 0:
            angle = angle * -1
        elif angle < 0 :
            angle = abs(angle)
        angle = int(angle)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img

    def create_chart(self, img, gas ,brake, lateBrakeArea, lateStartArea, followArea, dir_name, hardBreak, hardGas, speed, laneChange):
        fig= plt.figure(figsize=(14.20,1))
        axes= fig.add_axes([0.1,0.1,0.8,0.8])
        ax1 = plt.axes(frameon=False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        if laneChange[2] != '' and laneChange[3] != '':
            maxy = max(laneChange[1])
            y = np.array(laneChange[1])
            x = list(range(1,len(laneChange[1])+1))
            x = np.array(x)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = make_interp_spline(x, y)(x_smooth)
            line1, = plt.plot(x_smooth, y_smooth,color="green",label='Wheel')
            ax1.axhline(0,color='black',lw=1,zorder=0)
            plt.legend(handles=[line1],  loc='upper right')
            y_min = abs(min(y))
            y_max = abs(max(y))
            y_min_max = max(y_min,y_max)
            plt.text(laneChange[2],y_min_max-5,'Lane violation')
            ax1.axvspan(laneChange[2],laneChange[3], alpha=0.5, facecolor='y')
            plt.ylim(-y_min_max,y_min_max)
        else:
            y = np.array(gas)
            x = list(range(1,len(gas)+1))
            x = np.array(x)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = make_interp_spline(x, y)(x_smooth)
            
            if lateBrakeArea[0] != '' and lateBrakeArea[1] != '':
                plt.text(lateBrakeArea[0],85,'Late brake')
                ax1.axvspan(lateBrakeArea[0],lateBrakeArea[1], alpha=0.5, facecolor='r')
            if lateStartArea[0] != '' and lateStartArea[1] != '':
                plt.text(lateStartArea[0],85,'Late start')   
                ax1.axvspan(lateStartArea[0], lateStartArea[1], alpha=0.5, facecolor='g')       
            line1, = plt.plot(x_smooth, y_smooth,color="green",label='Gas')
            y = np.array(brake)
            x = list(range(1,len(brake)+1))
            x = np.array(x)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = make_interp_spline(x, y)(x_smooth)
            line2, = plt.plot(x_smooth, y_smooth,color="red",label='Brake')
            maxy = 100
            if followArea[0] != '' and followArea[1] != '':
                maxlist = brake + gas + speed
                maxy = max(maxlist)
                plt.text(followArea[0],maxy-5,'Follow too close')
                y = np.array(speed)
                x = list(range(1,len(speed)+1))
                x = np.array(x)
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = make_interp_spline(x, y)(x_smooth)
                line3, = plt.plot(x_smooth, y_smooth,color="yellow",label='Speed')
                ax1.axvspan(followArea[0], followArea[1], alpha=0.5, facecolor='y')
                
            if 'line3' in locals():
                plt.legend(handles=[line1,line2,line3],  loc='upper right')
            else:
                plt.legend(handles=[line1,line2],  loc='upper right')
            plt.ylim(-5,maxy)
            for i in range(len(hardBreak)):
                plt.text(hardBreak[i]-10,85,'Hard brake')
                ax1.axvspan(hardBreak[i]-10, hardBreak[i]+10, alpha=0.5, facecolor='y')
            for i in range(len(hardGas)):
                plt.text(hardGas[i]-10,85,'Hard gas')
                ax1.axvspan(hardGas[i]-10, hardGas[i]+10, alpha=0.5, facecolor='y')
        plt.axvline(x=int(img),color="blue",zorder=10)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        fig.tight_layout(pad=0)
        plt.savefig(dir_name+'/chart/image_'+img+'.png',pad_inches = 0)
        fig.clf()
        plt.close('all')

    def millions(SELF, x, pos):
            return pos

    def create_gas_bar(self, image, gas):
        x = np.arange(1)
        money = [gas]
        formatter = FuncFormatter(self.millions)
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(0.5, 1))
        plt.bar(x, money,color='green')
        fig.patch.set_facecolor('gray')
        ax.yaxis.set_major_formatter(formatter)
        ax.margins(x=0)
        plt.xticks(x)
        plt.ylim(0,100)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig(image,facecolor=fig.get_facecolor())
        fig.clf()
        plt.close('all')

    def create_brake_bar(self, image, brake):
        x = np.arange(1)
        money = [brake]
        formatter = FuncFormatter(self.millions)
        fig, ax = plt.subplots()
        fig = plt.figure(figsize=(0.5, 1))
        plt.bar(x, money,color='red')
        fig.patch.set_facecolor('gray')
        ax.yaxis.set_major_formatter(formatter)
        plt.xticks(x)
        plt.ylim(0,100)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig(image,facecolor=fig.get_facecolor())
        fig.clf()
        plt.close('all')

    def get_hard_brake(self, frames):
        cur = self.db['ml']
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incident';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        hard_brake = list()
        if table_exist == 1:
            j = 0
            for frame in frames:
                cur.execute("SELECT frame_num FROM incident WHERE camera_id=? and incident=? and frame_num=?",(1,'hard_brake',frame))
                t = cur.fetchall()
                if len(t) > 0:
                    hard_brake.append(j)
                j = j + 1
        return hard_brake

    def get_hard_gas(self, frames):
        cur = self.db['ml']
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incident';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        hard_gas = list()
        if table_exist == 1:
            j = 0
            for frame in frames:
                cur.execute("SELECT frame_num FROM incident WHERE camera_id=? and incident=? and frame_num=?",(1,'hard_gas',frame))
                t = cur.fetchall()
                if len(t) > 0:
                    hard_gas.append(j)
                j = j + 1
        return hard_gas

    def make_empty_video(self, size=(640,480), duration=5, fps=15, color=(0,0,0)):
        self.logger.info('Making empty video')
        ColorClip(size, color, duration=duration).write_videofile(self.recording_path+'out.mp4', fps=fps,logger=None)

    def remove_tmp_files(self):
        os.system('rm -rf tmp/*')

    def cover(self, image, small_image, loc):
        h, w, i = small_image.shape
        new_x1 = loc[0]
        new_x2 = loc[0] + (w)
        new_y1 = loc[1]
        new_y2 = loc[1] + (h)
        image[new_y1:new_y2, new_x1:new_x2, :] = small_image
        return image

    def create_center_info_column(self, wheel, gas, brake, speed, left_turn, right_turn):
        cur_img = np.zeros((480, 140, 3), dtype=np.uint8)

        # Logo
        #=======
        self.cover(cur_img, self.icons["foresight_logo"], [40,1])

        # Brake
        #=======
        self.cover(cur_img, self.icons["red_pedal"], [17,71])
        brake_bar_img = self.create_pedal_bar(brake, self.red)
        self.cover(cur_img, brake_bar_img, [17,127])

        # Gas
        #=======
        self.cover(cur_img, self.icons["green_pedal"], [80,71])
        gas_bar_img = self.create_pedal_bar(gas, self.green)
        self.cover(cur_img, gas_bar_img, [80,127])

        # Speed and Wheel Icons
        # =====================
        self.cover(cur_img, self.icons["speed"], [17,210])
        wheel_image = self.rotation(self.icons["wheel"], wheel)
        cur_img = self.cover(cur_img, wheel_image, [17,319])

        # Speed and Wheel Text Elements
        #==============================
        img_pil = Image.fromarray(cur_img)
        draw = ImageDraw.Draw(img_pil)
        center_x = 70

        # speed kmh
        x = 62
        y = 250
        num_chars = len(str(speed))
        if num_chars == 2:
            x -= 7
        elif num_chars >= 3:
            x -= 14

        draw.text((x, y), str(speed), font=self.speed_font, fill=self.green, anchor='ls')
        draw.text((center_x, y+14),  "km/h", font=self.speed_unit_font, fill=self.green, anchor='ms')

        # speed mph
        speed_mph = int(speed * 0.621371)
        x = 62
        y = y+45
        num_chars = len(str(speed_mph))
        if num_chars == 2:
            x -= 7
        elif num_chars >= 3:
            x -= 14
        draw.text((x, y), str(speed_mph), font=self.speed_font, fill=self.red, anchor='ls')
        draw.text((center_x, y+14),  "mph", font=self.speed_unit_font, fill=self.red, anchor='ms')

        # wheel angle
        x = 65
        y = 378
        wheel_str = str(wheel)
        if wheel > 0:
            wheel_str = "+" + str(wheel);
        num_chars = len(wheel_str)
        if num_chars == 2:
            x -= 7
        elif num_chars == 3:
            x -= 12
        elif num_chars == 4:
            x -= 16
        draw.text((x, y), wheel_str, font=self.wheel_font, fill=self.green, anchor='ls')

        cur_img = np.array(img_pil)

        # Turn Signal
        # ===========
        y = 433
        x1 = 17
        x2 = 80
        if left_turn:
            self.cover(cur_img, self.icons["left_indicator_on"], [x1,y])
        else:
            self.cover(cur_img, self.icons["left_indicator_off"], [x1,y])

        if right_turn:
            self.cover(cur_img, self.icons["right_indicator_on"], [x2,y])
        else:
            self.cover(cur_img, self.icons["right_indicator_off"], [x2,y])

        return cur_img

    def get_lane_change(self, frames):
        cur = self.db['ml']
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incident';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        lane_change = list()
        if table_exist == 1:
            lane_change = list()
            i = 0
            for frame in frames:
                cur.execute("SELECT frame_num FROM incident where frame_num=? and incident=?",(frame,'lane_change_no_signal',))
                res = cur.fetchone()
                if res is not None:
                    lane_change.append(i)
                i = i + 1
        return lane_change

    def get_wheel_prediction(self):
        cur = self.db['ml']
        wheel = list()
        cur.execute("SELECT CAST(final_output AS INT) FROM lane_detection_model where camera_id=?",(1,))
        wheel = cur.fetchall()
        return wheel

    def convert_date(self, start_date, sec):
        res = int(start_date) + (int(sec) // 1000000000)
        res = datetime.datetime.fromtimestamp(res).strftime("%Y/%m/%d %H:%M:%S")
        return res

    def get_recording_name(self):
        recording_name = self.video_name.split('/')[-1].replace('.zip', '')
        return recording_name

    def get_gps(self):
        cur = self.db['gps']
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gps_latlng';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        loc = list()
        if table_exist == 1:
            cur.execute("SELECT lat,lng FROM gps_latlng")
            loc = cur.fetchall()
            if len(loc) > 0:
                lat = list()
                lng = list()
                for l in loc:
                    if l[0] != None and [1] != None:
                        lat.append(l[0])
                        lng.append(l[1])
                if len(lat) > 0 or len(lng) > 0:
                    location = [sum(lat) / len(lat),sum(lng) / len(lng)]
                else:
                    location = False
            else:
                location = False
        else:
            location = None
        return location

    def find_day_night(self):
        cur = self.db['ml']
        cur.execute("SELECT day_night FROM metadata")
        day_night = cur.fetchone()
        return day_night[0]

    def version_cheker(self, dbr, col):
        cur = self.db[dbr]
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        if table_exist == 1:
            cur.execute("SELECT "+ col +" FROM metadata")
            version = cur.fetchone()
            res = version[0]
        else:
            res = None
        return res

    def get_red_pass(self):
        cur = self.db['ml']
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incident';")
        table_exist = cur.fetchall()
        table_exist = len(table_exist)
        red_pass = list()
        if table_exist == 1:
            cur.execute("SELECT frame_num FROM incident WHERE camera_id=? and incident=?",(1,'red_pass'))
            res = cur.fetchall()
            red_pass.append(res)
        else:
            red_pass = list()
        return red_pass

    def closest(self, lst, K):
        lst = np.asarray(lst)
        idx = (np.abs(lst - K)).argmin()
        return lst[idx]

    def run(self, file_path):
        if file_path == False:
            return False
        self.recording_path = file_path
        destination_path = self.recording_path+"/final_video.mp4"
        self.recording_name = self.recording_path
        self.recording_path = self.recording_path + '/'
        skip_icons = True
        skip_time = True
        skip_front_center_narrow = True
        front_center_narrow_size = 0
        skip_interior_center = True
        interior_center_size = 0
        self.recording_name = self.get_recording_name()

        # Check vehicle and ml databases exist
        if os.path.exists(self.recording_path+'ml.db') == True and os.path.exists(self.recording_path+'recording.db') == True:
            skip_icons = False
        if os.path.exists(self.recording_path+'ml.db') == False:
            self.messages.append('ml.db')
        if os.path.exists(self.recording_path+'recording.db') == False:
            self.messages.append('recording.db')

        # Check that gps database exists
        if  os.path.exists(self.recording_path+'gps.db') == True:
            skip_time = False
        else:
            self.messages.append('gps.db')

        # Check that front facing video exists
        if os.path.exists(self.recording_path+'front_center_narrow.mkv') == True :
            skip_front_center_narrow = False
            front_center_narrow_size = os.stat(self.recording_path+'front_center_narrow.mkv').st_size
        else:
            self.messages.append('front_center_narrow.mkv')

        # Check that interior facing video exists
        if os.path.exists(self.recording_path+'interior_center.mkv') == True :
            skip_interior_center = False
            interior_center_size = os.stat(self.recording_path+'interior_center.mkv').st_size
        else:
            self.messages.append('interior_center.mkv')
        
        self.logger.info('Checking Files')
        try:
            self.get_db_data('ml')
            self.get_db_data('gps')
            self.get_db_data('recording')
            version = self.version_cheker('ml', 'recording_version').split('.')
            mlstreamer_split = self.mlstreamer_versioon.split('.')
            if (mlstreamer_split[0] != version[0]):
                self.logger.error('mlstreamer version error')
                self.messages.append('ML streamer version error')
                sys.exit()
            elif (mlstreamer_split[1] != version[1]):
                self.logger.warning('mlstreamer minor error')
            elif (mlstreamer_split[2] != mlstreamer_split[2]):
                self.logger.warning('recording path error')
            version = ''
            version = self.version_cheker('recording', 'version').split('.')
            recording_split = self.recording_version.split('.')
            if (recording_split[0] != version[0]):
                self.logger.error('recording version error')
                self.messages.append('recording version error')
                sys.exit()
            elif (recording_split[1] != recording_split[1]):
                self.logger.warning('recording minor error')
            elif (recording_split[2] != recording_split[2]):
                self.logger.warning('recording path error')
            if skip_icons == False:
                frames = self.get_frames()
                frameNums = list()
                if frames == True:
                    skip_icons = True
                for frame in frames:
                    frameNums.append(frame[1])
                gas = self.get_gas()
                wheel = self.get_wheel(frames)
                speed = self.get_speed()
                brake = self.get_brake()
                leftIndicator = self.get_left_indicator()
                rightIndicator = self.get_right_indicator()
                dayNight = self.find_day_night()
                if dayNight != 'day':
                    skip_traffic_light = True

            #Make internal video
            if skip_interior_center == False and interior_center_size > 0:
                interior_cam = cv2.VideoCapture(self.recording_path+"interior_center.mkv")
                skip_traffic_light = False
            else:
                interior_cam = None
                skip_traffic_light = True

            #Make external video
            if skip_front_center_narrow == False and front_center_narrow_size > 0:
                exterior_cam = cv2.VideoCapture(self.recording_path+"front_center_narrow.mkv")
            else:
                exterior_cam = None
            
            if exterior_cam is None and interior_cam is None:
                self.logger.warning('Making black video')
                self.make_empty_video((1450, 480), 5)
            #Combine 2 videos
            self.logger.info('Combining images')
            if skip_icons == False and (exterior_cam is not None or interior_cam is not None):
                if skip_time == False:
                    recordingDate = self.recording_name
                    year = recordingDate[0:4]
                    month = recordingDate[4:6]
                    day = recordingDate[6:8]
                    hour = recordingDate[8:10]
                    minute = recordingDate[10:12]
                    second = recordingDate[12:14]
                    startTime = year+'/'+month+'/'+day+' '+hour+':'+minute+':'+second
                    location = self.get_gps()
                    if location == False or location == None:
                        skip_conert_time = True
                    else :
                        skip_conert_time = False
                    if skip_conert_time == False:
                        tf = TimezoneFinder()
                        latitude, longitude = location[0], location[1]
                        to_zone = tf.timezone_at(lng=longitude, lat=latitude)
                        from_zone = tz.gettz('UTC')
                        to_zone = tz.gettz(to_zone)
                        utc = datetime.datetime.strptime(startTime, '%Y/%m/%d %H:%M:%S')
                        utc = utc.replace(tzinfo=from_zone)
                        startTime = utc.astimezone(to_zone)
                        startTime = time.mktime(startTime.timetuple())
                    else :
                        utc_datetime = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), 0, tzinfo=datetime.timezone.utc)
                        local_timezone = pytz.timezone("America/Toronto")
                        local_datetime = utc_datetime.replace(tzinfo=pytz.utc)
                        startTime = local_datetime.astimezone(local_timezone)
                        startTime = time.mktime(startTime.timetuple())
                i = 0
                out = cv2.VideoWriter(self.recording_path+'out.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1420,480))

                # Middle frame taken from either interior or exterior frame count.
                # 0 just in case
                middle_frame = 0
                if exterior_cam is not None:
                    middle_frame = int(exterior_cam.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
                elif interior_cam is not None:
                    middle_frame = int(interior_cam.get(cv2.CAP_PROP_FRAME_COUNT) / 2)

                #Main loop
                while(True):
                    # As long as one of the cameras gives us frames, continue 
                    ext_ret = False
                    if exterior_cam is not None:
                        ext_ret, ext_frame = exterior_cam.read()
                        if not ext_ret:
                            exterior_cam.release()
                            exterior_cam = None

                    int_ret = False
                    if interior_cam is not None:
                        int_ret, int_frame = interior_cam.read()
                        if not int_ret:
                            interior_cam.release()
                            interior_cam = None
                    
                    # Both cameras have stopped giving us frames
                    if exterior_cam is None and interior_cam is None:
                        break
                    
                    # Combine exterior and interior frames against black background
                    raw_frame = np.zeros((480, 1420, 3), dtype=np.uint8)
                    if ext_ret:
                        raw_frame = self.cover(raw_frame, ext_frame, [780, 0])
                    if int_ret:
                        raw_frame = self.cover(raw_frame, int_frame, [0, 0])

                    #wheel
                    cur_wheel = wheel[i] if i < len(wheel) else 0  
                    cur_gas = gas[i] if i < len(gas) else 0  
                    cur_brake = brake[i] if i < len(brake) else 0  
                    cur_speed = speed[i] if i < len(speed) else 0  
                    cur_left_indicator = leftIndicator[i] if i < len(leftIndicator) else 0
                    cur_right_indicator = rightIndicator[i] if i < len(rightIndicator) else 0
                    center_info_column = self.create_center_info_column(
                        wheel=cur_wheel,
                        gas=cur_gas,
                        brake=cur_brake,
                        speed=cur_speed,
                        left_turn=cur_left_indicator,
                        right_turn=cur_right_indicator
                    )
                    newImg = self.cover(raw_frame, center_info_column, [640,0])
                    
                    #traffic light
                    skip_traffic_light = True
                    if skip_traffic_light == False:
                        frameLight = self.findTrafficLight(self.recording_path+'/ml.db',frameNums[i])
                        if len(frameLight) > 0:
                            x = frameLight[0][1] + 780
                            y = frameLight[0][2]
                            if frameLight[0][0] != 0:
                                if frameLight[0][0] == 1:
                                    color = self.green_light
                                elif frameLight[0][0] == 2:
                                    color = self.red_light
                                center_coordinates = (x,y)
                                newImg = cv2.circle(newImg, center_coordinates, self.traffic_light_circle_size, color, -1)

                    #is interior camera covered
                    if skip_interior_center:
                        self.cover(newImg, self.icons["interior_covered"], [220,190])

                    #write date time
                    if skip_time == False:
                        position = (10,470)
                        totalTime = frames[i][0] if i < len(frames) else frames[-1][0]
                        frameDateTime = self.convert_date(startTime, totalTime)
                        cv2.putText(newImg, str(frameDateTime), position,cv2.FONT_HERSHEY_SIMPLEX, self.date_time_font_size, self.date_time_color, 2)

                    #write recording id
                    position = (250,470)
                    cv2.putText(newImg, str(self.recording_name), position,cv2.FONT_HERSHEY_SIMPLEX, self.recording_name_font_size, self.recording_name_color, 2)

                    out.write(newImg)

                    if i == middle_frame:
                        cv2.imwrite(self.recording_path + 'thumbnail.jpg', newImg,  [cv2.IMWRITE_JPEG_QUALITY, 9])

                    i = i + 1

                out.release()
                if exterior_cam is not None:
                    exterior_cam.release()
                if interior_cam is not None:
                    interior_cam.release()
        except:
            self.make_empty_video((1450, 480), 5)
            self.logger.exception(f"Error in video merger : {self.recording_name}")

        self.logger.info('Making final video')
        clip1 = VideoFileClip(self.recording_path+'out.mp4')
        clip1.write_videofile(destination_path,logger=None)
        if (os.path.exists(file_path+'/microphone.ogg') == True and os.stat(file_path+'/microphone.ogg').st_size > 0):
            input_video = ffmpeg.input(destination_path)
            input_audio = ffmpeg.input(self.recording_path+'microphone.ogg')
            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(self.recording_path+'FinalWithAudio.mp4').run()
            os.system('rm -rf ' + destination_path)
            os.system('mv '+ self.recording_path+'FinalWithAudio.mp4' + ' ' + destination_path)

        if self.aws_id is not None and self.aws_upload_path is not None:
            self.logger.info('Removing tmp files')
            os.system('rm -rf ' + self.recording_path)

if __name__ == '__main__':
    video = video_merger(sys.argv[1])
    video.run(sys.argv[1])
