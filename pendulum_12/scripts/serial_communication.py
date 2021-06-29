#!/usr/bin/env python3


from numpy import DataSource, ma
import rospy
#import threading
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
#import gym
import math
from std_srvs.srv import Empty
import time
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState

data_received = np.zeros(4)

flag  = False
count = 0

class env:
	def __init__(self):
		rospy.init_node('talker', anonymous = True)

		self.pub = rospy.Publisher('/ip/joint_controller/command', Float64, queue_size = 10) 
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.set_link = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)

	def step(self,action):
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except:
			pass
		self.pub.publish(action)
	#get data
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/ip/joint_states', JointState, timeout=5)
			except:
				pass
	#take action
		
		#pause simulation
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except:
			pass
		return data
	
	def reset(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/unpause_physics service call failed")
		
		rospy.wait_for_service('/gazebo/set_link_state')
		self.set_link(LinkState(link_name='arm'))
		self.set_link(LinkState(link_name='pendulum'))
	
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except (rospy.ServiceException) as e:
			print ("/gazebo/pause_physics service call failed")
		
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/ip/joint_states', JointState, timeout=5)
			except:
				pass
		return data






gazebo = env()
#rospy.Subscriber('/ip/joint_states', JointState, callback)
while not rospy.is_shutdown():
	#unpause simulation
	data = gazebo.step(0.13)
	if(count>20):
		print('....reseting env.....')
		data = gazebo.reset()
		time.sleep(5)
		count=0
	print('data_received_after_normalization = {},count ={}'.format(data.position,count))
	count+=1
