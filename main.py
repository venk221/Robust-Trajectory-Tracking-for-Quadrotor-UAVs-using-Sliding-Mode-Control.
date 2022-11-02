#!/usr/bin/env python3
from math import pi, sqrt, atan2, cos, sin,asin
from turtle import position
import numpy as np
from numpy import NaN
from numpy import linspace
import rospy
import tf
from std_msgs.msg import Empty, Float32
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Twist, Pose2D
import pickle
import os

from Quintic_Trajectory import Quintic_Trajectory

class Quadrotor():
    def __init__(self):
        # publisher for rotor speeds
        self.motor_speed_pub = rospy.Publisher("/crazyflie2/command/motor_speed", Actuators, queue_size=10)
        
        # subscribe to Odometry topic
        self.odom_sub = rospy.Subscriber("/crazyflie2/ground_truth/odometry",Odometry, self.odom_callback)
        
        self.t0 = None
        self.t = None
        self.t_series = []
        self.x_series = []
        self.y_series = []
        self.z_series = []
        self.mutex_lock_on = False
        rospy.on_shutdown(self.save_data)
        
        # TODO: include initialization codes if needed
        self.des_pos=list()
        self.des_vel =list()
        self.des_acc =list()
        self.des_rp = list()

        # Physical parameters
        self.m = 0.027
        self.l = 0.046
        self.g = 9.81
        self.Ip = 12.65625e-8
        self.Ix = 16.571710e-6
        self.Iy=16.571710e-6
        self.Iz = 29.261652e-6
        self.omega_max = 2618
        self.omega_min = 0
        self.Kf = 1.28192e-8
        self.Km = 5.964552e-3
        self.big_O =0

    def traj_evaluate(self):
        # TODO: evaluating the corresponding trajectories designed in Part 1 to return the desired positions, velocities and accelerations
        traj_obj = Quintic_Trajectory()
        self.des_pos = traj_obj.get_position(self.t)
        self.des_vel = traj_obj.get_velocities(self.t)
        self.des_acc = traj_obj.get_acclerations(self.t)


    def convert_force_to_rotor_speed(self,forces):
        ''' Takes input as u = [u1,u2,u3,u4] and returns rotor speeds (w1,w2,w3,w4) calculated using allocation matrix'''
        u1 = forces[0]
        u2 = forces[1]
        u3 = forces[2]
        u4 = forces[3]
        w1_sqr=u1/(4*self.Kf)-(sqrt(2)*u2)/(4*self.Kf*self.l)-(sqrt(2)*u3)/(4*self.Kf*self.l)-u4/(4*self.Km*self.Kf)
        w2_sqr=u1/(4*self.Kf)-(sqrt(2)*u2)/(4*self.Kf*self.l)+(sqrt(2)*u3)/(4*self.Kf*self.l)+u4/(4*self.Km*self.Kf)
        w3_sqr=u1/(4*self.Kf)+(sqrt(2)*u2)/(4*self.Kf*self.l)+(sqrt(2)*u3)/(4*self.Kf*self.l)-u4/(4*self.Km*self.Kf)
        w4_sqr=u1/(4*self.Kf)+(sqrt(2)*u2)/(4*self.Kf*self.l)-(sqrt(2)*u3)/(4*self.Kf*self.l)+u4/(4*self.Km*self.Kf)
        sqrs= [w1_sqr,w2_sqr,w3_sqr,w4_sqr]
        for idx in range(len(sqrs)):
            if sqrs[idx] >= 2618*2618:
                sqrs[idx]= 2618*2618
            elif sqrs[idx] <= 0:
                sqrs[idx] = 0
        w1 = sqrt(sqrs[0])
        w2 = sqrt(sqrs[1])
        w3 = sqrt(sqrs[2])
        w4 = sqrt(sqrs[3])
        return [w1,w2,w3,w4]

    def get_des_rpy(self,x,y,dot_x,dot_y,u1):
        ########################################### Tuning Parameter Kp, Kd
        self.Kp = 120
        self.Kd = 10
        ###########################################
        x_d = self.des_pos[0]
        y_d = self.des_pos[1]
        dot_x_d = self.des_vel[0]
        dot_y_d = self.des_vel[1]
        ddot_x_d = self.des_acc[0]
        ddot_y_d = self.des_acc[1]
        Fx = self.m*(-self.Kp*(x-x_d)-self.Kd*(dot_x-dot_x_d)+ddot_x_d)
        Fy = self.m*(-self.Kp*(y-y_d)-self.Kd*(dot_y-dot_y_d)+ddot_y_d)
        if abs(Fx/u1) > 1:
            val_1 = np.sign(Fx/u1)*1
        else:
            val_1 = Fx/u1
        theta_d = asin(val_1)
        if abs(Fy/u1) >1:
            val_2 = np.sign(Fy/u1)*1
        else:
            val_2 = Fy/u1
        phi_d = asin(-val_2)
        self.des_rp = [theta_d,phi_d]
        return self.des_rp
    
    def saturation_fxn(self,s,phi_sat=0.1):
        return min(max(s/phi_sat,-1),1)

    def smc_control(self, xyz, xyz_dot, rpy, rpy_dot):
        '''
        Position
            xyz - [[x pos],[y pos],[z pos]]
            xyz_dot - [[x vel],[y vel],[z vel]]
        Orientation
            rpy - roll pitch yaw
            rpy_dot - 
        '''
        # obtain the desired values by evaluating the corresponding trajectories
        self.traj_evaluate()
        # TODO: implement the Sliding Mode Control laws designed in Part 2 to calculate the control inputs "u"
        z = xyz[2][0]
        phi = rpy[0][0]
        theta = rpy[1][0]
        psy = rpy[2][0]

        #err = np.array([z - self.des_pos[2],phi - self.des_rp[0],theta - self.des_rp[1],psy])
        dot_z = xyz_dot[2][0]
        dot_phi = rpy_dot[0][0]
        dot_theta = rpy_dot[1][0]
        dot_psy=rpy_dot[2][0]
        dot_err = np.array([dot_z - self.des_vel[2],dot_phi ,dot_theta,dot_psy])
        
        ### SMC Parameters -- Tuning parmaters
        lmd = 1
        phi_sat = 0.3
        
        k1 = 10
        k2 = 5
        k3 = 5
        k4 = 5
        #s = dot_err + lmd*err

        # For z 
        s1 = dot_z - self.des_vel[2] + lmd*(z-self.des_pos[2])
        sat_1 = self.saturation_fxn(s1,phi_sat)
        u_r1 = -k1*sat_1
        u1 = (self.m/(cos(phi)*cos(theta)))*(self.g+self.des_acc[2]-lmd*dot_err[0]+u_r1)

        des_phi_theta = self.get_des_rpy(xyz[0][0],xyz[1][0],xyz_dot[0][0],xyz_dot[1][0],u1)
        
        # REMARK: wrap the roll-pitch-yaw angle errors to [-pi to pi]
        err = [z-self.des_pos[2],phi - des_phi_theta[0], theta-des_phi_theta[1], psy]
        for e in err:
            if e > pi:
                e= pi
            elif e< -pi:
                e = -pi
        
        # For phi 
        s2 = dot_err[1] + lmd*err[1]
        sat_2 = self.saturation_fxn(s2,phi_sat)
        u_r2 = -k2*sat_2
        u2 = (self.Ix)*(-dot_theta*dot_psy*(self.Iy-self.Iz)/self.Ix + self.Ip*self.big_O*dot_theta/self.Ix - lmd*dot_err[1] + u_r2) 
        

        # For theta
        s3 = dot_err[2] + lmd*err[2]
        sat_3 = self.saturation_fxn(s3,phi_sat)
        u_r3 = -k3*sat_3
        u3 = (self.Iy)*(-dot_phi*dot_psy*(self.Iz-self.Ix)/self.Iy - self.Ip*self.big_O*dot_phi/self.Iy - lmd*dot_err[2] + u_r3) 
        
        # For psy
        s4 = dot_err[3] + lmd*err[3]
        sat_4 = self.saturation_fxn(s4,phi_sat)
        u_r4 = -k4*sat_4
        u4 = (self.Iz)*(-dot_phi*dot_theta*(self.Ix-self.Iy)/self.Iz - lmd*dot_err[3] + u_r4) 
        
        u = [u1,u2,u3,u4]

        # TODO: convert the desired control inputs "u" to desired rotor velocities "motor_vel" by using the "allocation matrix"
        motor_vel = self.convert_force_to_rotor_speed(u)

        # TODO: maintain the rotor velocities within the valid range of [0 to 2618]
        for vel in motor_vel:
            if vel > 2618:
                vel = 2618
            elif vel<0:
                vel = 0 
        self.big_O = motor_vel[0]-motor_vel[1]+motor_vel[2]-motor_vel[3]
        # publish the motor velocities to the associated ROS topic
        motor_speed = Actuators()
        motor_speed.angular_velocities = [motor_vel[0], motor_vel[1], motor_vel[2], motor_vel[3]]
        self.motor_speed_pub.publish(motor_speed)
        if motor_vel != [0,0,0,0] : 
            print("Motor speeds ", motor_vel)

    # odometry callback function (DO NOT MODIFY)
    def odom_callback(self, msg):
        if self.t0 == None:
            self.t0 = msg.header.stamp.to_sec()
        self.t = msg.header.stamp.to_sec() - self.t0
    
        # convert odometry data to xyz, xyz_dot, rpy, and rpy_dot
        w_b = np.asarray([[msg.twist.twist.angular.x], [msg.twist.twist.angular.y], [msg.twist.twist.angular.z]])
        v_b = np.asarray([[msg.twist.twist.linear.x], [msg.twist.twist.linear.y], [msg.twist.twist.linear.z]])
        xyz = np.asarray([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.pose.pose.position.z]])
        q = msg.pose.pose.orientation
        T = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0:3, 3] = xyz[0:3, 0]
        R = T[0:3, 0:3]
        xyz_dot = np.dot(R, v_b)
        rpy = tf.transformations.euler_from_matrix(R, 'sxyz')
        rpy_dot = np.dot(np.asarray([
        [1, np.sin(rpy[0])*np.tan(rpy[1]), np.cos(rpy[0])*np.tan(rpy[1])],
        [0, np.cos(rpy[0]), -np.sin(rpy[0])],
        [0, np.sin(rpy[0])/np.cos(rpy[1]), np.cos(rpy[0])/np.cos(rpy[1])]
        ]), w_b)
        rpy = np.expand_dims(rpy, axis=1)

        # store the actual trajectory to be visualized later
        if (self.mutex_lock_on is not True):
            self.t_series.append(self.t)
            self.x_series.append(xyz[0, 0])
            self.y_series.append(xyz[1, 0])
            self.z_series.append(xyz[2, 0])
        # call the controller with the current states
        self.smc_control(xyz, xyz_dot, rpy, rpy_dot)

    # save the actual trajectory data
    def save_data(self):
        # TODO: update the path below with the correct path
        with open("log.pkl","wb") as fp:
            self.mutex_lock_on = True
            pickle.dump([self.t_series,self.x_series,self.y_series,self.z_series], fp)
    
if __name__ == '__main__':
    rospy.init_node("quadrotor_control")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = Quadrotor()
    try:
        rospy.spin()

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    