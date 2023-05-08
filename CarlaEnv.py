#!/usr/bin/env python
from __future__ import print_function
import glob
import os
import sys
import time
import numpy as np
import cv2
import math
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import random
sys.path.append('C:/Users/ruski/Desktop/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
SHOW_PREVIEW = False

IMG_WIDTH=64
IMG_HEIGHT=64
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0

    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    actor_list = []

    front_camera = None
    collision_hist = []
    SECONDS_PER_EPISODE=10

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)

    def reset(self):
        self.num_states=0
        self.collision_hist = []
        self.actor_list = []

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.load_world('Town01')
        time.sleep(1.5)

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        #print(blueprint_library.filter('vehicle'))
        self.model_3 = blueprint_library.filter('model3')[0]
        amap = self.world.get_map()
        sampling_resolution = 1.05
        # dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        # grp.setup()
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.randint(len(spawn_points))
        end_point = np.random.randint(len(spawn_points))
        while spawn_point==end_point:
            spawn_point = np.random.randint(len(spawn_points))
            end_point = np.random.randint(len(spawn_points))
        a = carla.Location(spawn_points[167].location)
        b = carla.Location(spawn_points[254].location)#189,254
        self.w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
        i = 0
        for i in range(len(self.w1)-1):
            self.world.debug.draw_line(self.w1[i][0].transform.location + carla.Location(z=0.25),self.w1[i+1][0].transform.location + carla.Location(z=0.25),thickness=0.1)
        # for i in range(len(self.w1)):
        #     self.world.debug.draw_string(self.w1[i][0].transform.location + carla.Location(z=0.25),"0",life_time=1000.0,persistent_lines=True)
        # for i in range(len(spawn_points)):
        #     print(spawn_points[i])
        #     self.world.debug.draw_string(spawn_points[i].location,f'{i}',life_time=1000.0,persistent_lines=True)
        #
        # time.sleep(50)


        self.transform = self.world.get_map().get_spawn_points()[167]
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')
        # self.rgb_cam.set_attribute('sensor_tick', '0.1')


        transform = carla.Transform(carla.Location(z=10), carla.Rotation(pitch=-90))

        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
        # self.vehicle.set_autopilot(True)

        return self.front_camera,np.zeros(11)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        self.num_states+=1
        # cv2.imshow("",i3)
        # cv2.waitKey(1)
        self.front_camera = i3/255

    def step(self, action):
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=float(action)))
        # if action == 0: #forward-slow
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        # if action == 1: #left-slow
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.25*self.STEER_AMT))
        # if action == 2: #left-medium
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.5*self.STEER_AMT))
        # if action == 3: #left-hard
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1*self.STEER_AMT))
        # if action == 4: #right-slow
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.25*self.STEER_AMT))
        # if action == 5: #right-medium
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5*self.STEER_AMT))
        # if action == 6: #right-hard
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1*self.STEER_AMT))


        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            done = False
            reward = 20


        dist_list=[]
        car_location=self.vehicle.get_location()
        dist_from_start=car_location.distance(self.w1[0][0].transform.location)
        # reward+=dist_from_start

        for i,w in enumerate(self.w1):
            w_location=w[0].transform.location
            dist=car_location.distance(w_location)
            dist_list.append((dist,w_location,i))

        dist_list=sorted(dist_list, key=lambda x: x[0])
        closest_index=dist_list[0][2]
        # total_angle=0
        angle_list=[]
        for i in range(closest_index+1,min(closest_index+12,len(self.w1))):
            point1=self.w1[i][0].transform.location
            angle_list.append(self.get_angle(self.vehicle,point1))
            # total_angle+=self.get_angle(self.vehicle,point1,point2)
        if dist_list[0][0]>5:
            done=True
            reward=-100
        # reward-=dist_list[0][0]
        # if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
        #     done = True
        if(len(angle_list)<11):
            diff=11-len(angle_list)
            angle_list=np.pad(angle_list, (0, diff), mode='constant')

        return self.front_camera, reward, done, np.array(angle_list)/180

    def get_angle(self,vehicle ,point1):
        vehicle_location=vehicle.get_transform().location
        v1 = np.array([point1.x, point1.y]) - np.array([vehicle_location.x, vehicle_location.y])

        # Get the forward vector of the vehicle
        forward_vector = [vehicle.get_transform().get_forward_vector().x,vehicle.get_transform().get_forward_vector().y]

        # Calculate the dot product between the two vectors
        dot_product = np.dot(forward_vector, v1)

        # Calculate the magnitudes of both vectors
        magnitude1 = np.linalg.norm(forward_vector)
        magnitude2 = np.linalg.norm(v1)

        # Calculate the angle between the two vectors
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))

        signed_angle, angle = np.arctan2(np.cross(v1, forward_vector), np.dot(forward_vector, v1)) * 180 / np.pi, angle * 180 / np.pi

        # Convert the angle to degrees
        # angle_degrees = np.degrees(angle)
        # v2=np.array([-point1.x+vehicle.get_transform().location.x,-point1.y+vehicle.get_transform().location.y])
        # dot_product = np.dot(v1, v2)
        # magnitude_v1 = np.linalg.norm(v1)
        # magnitude_v2 = np.linalg.norm(v2)
        # angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
        # angle_degrees = np.degrees(angle)
        return signed_angle

        # vehicle_position=vehicle.get_transform().location
        # vehicle_yaw=180-vehicle.get_transform().rotation.yaw
        #
        # point1_x = point1.x
        # point1_y = point1.y
        # point2_x = point2.x
        # point2_y = point2.y
        #
        # # Compute the slope of the line formed by the two points
        # # if abs(point2_x - point1_x)<1e-4:
        # #     line_angle=math.pi/2
        # try:
        #     slope = (point2_y - point1_y) / (point2_x - point1_x)
        #     line_angle = math.atan(slope)
        # except:
        #     print(point1_x,point1_y,point2_x,point2_y)
        #     line_angle=math.pi/2
        #
        # # Compute the angle of the line with respect to the x-axis
        #
        #
        # line_degrees=math.degrees(line_angle)
        #
        # # Convert the yaw angle of the car to radians
        # vehicle_yaw_radians = math.radians(vehicle_yaw)
        #
        # # Compute the angle between the car's yaw and the line
        # angle_diff = vehicle_yaw_radians - line_angle
        #
        # # Convert the angle difference to degrees
        # angle_diff_degrees = math.degrees(angle_diff)
        # return angle_diff_degrees

        # rotation_matrix = np.array([
        #     [np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0],
        #     [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0],
        #     [0, 0, 1],
        # ])
        # forward_direction_vector = np.dot(rotation_matrix, np.array([1, 0, 0]))
        # forward_unit_vector = forward_direction_vector / np.linalg.norm(forward_direction_vector)
        # angle_between_vectors = np.arccos(np.dot(np.array([direction_unit_vector.x,direction_unit_vector.y,direction_unit_vector.z]), forward_unit_vector))
        # return math.degrees(angle_between_vectors)
        # vehicle_rotation = vehicle.get_transform().rotation.yaw
        #
        # # Get the angle between the car's pose and the line formed by the two points
        # delta_x = point2.x - point1.x
        # delta_y = point2.y - point1.y
        # target_rotation = math.degrees(math.atan2(delta_y, delta_x))
        #
        # angle = target_rotation - vehicle_rotation
        # return angle
