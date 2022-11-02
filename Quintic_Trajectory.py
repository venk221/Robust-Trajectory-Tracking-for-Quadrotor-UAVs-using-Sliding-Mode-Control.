import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from collections import defaultdict

np.set_printoptions(suppress=True)

class Quintic_Trajectory():
    def __init__(self):
        self.num_waypoints = 5
        self.waypoints = [(0,0,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1),(0,0,1)]
        self.traj_x=[]
        self.traj_y=[]
        self.traj_z=[]
        self.traj_x_vel=[]
        self.traj_y_vel=[]
        self.traj_z_vel=[]
        self.traj_x_acc=[]
        self.traj_y_acc=[]
        self.traj_z_acc=[]
        self.full_trajectory = {'pos':[self.traj_x,self.traj_y,self.traj_z],'vel':[self.traj_x_vel,self.traj_y_vel,self.traj_z_vel],'acc':[self.traj_x_acc,self.traj_y_acc,self.traj_z_acc]}
        self.time_breaks=[0,5,20,35,50,65]
        self.coefficients=defaultdict()

    def get_coefficients(self,t_init,t_final,init_params,final_params):
        init_pos = init_params[0]
        init_vel = 0
        init_acc = 0
        final_pos = final_params[0]
        final_vel = 0
        final_acc = 0
        A = np.array([[1,t_init,t_init**2,t_init**3,t_init**4,t_init**5],[0,1,2*t_init,3*t_init**2,4*t_init**3,5*t_init**4],[0,0,2,6*t_init,12*t_init**2,20*t_init**3],[1,t_final,t_final**2,t_final**3,t_final**4,t_final**5],[0,1,2*t_final,3*t_final**2,4*t_final**3,5*t_final**4],[0,0,2,6*t_final,12*t_final**2,20*t_final**3]])
        B = np.array([init_pos,init_vel,init_acc,final_pos,final_vel,final_acc])

        coeffs = np.linalg.solve(A,B) # Solves for x in Ax=B
        return coeffs

    def traj_eq_coeff(self,t):
        ''' Returns Coefficient for each coordinates - x,y,z
            c_x = [ a1, a2, a3, a4, a5, a6]
            c_y = [ a1, a2, a3, a4, a5, a6]
            c_z = [ a1, a2, a3, a4, a5, a6]
        '''
        if t >= 0 and t<5:          # Waypoint 1 to 2 (0,0,0) to (0,0,1) in 5 sec
            c_x = self.get_coefficients(0,5,[0,0,0],[0,0,0])    # X coord 0 to 0
            c_y = self.get_coefficients(0,5,[0,0,0],[0,0,0])    # Y coord 0 to 0
            c_z = self.get_coefficients(0,5,[0,0,0],[1,0,0])    # Z coord 0 to 1
        elif t>=5 and t<20:         # Waypoint 2 to 3 (0,0,1) to (1,0,1) in 15 sec
            c_x = self.get_coefficients(0,15,[0,0,0],[1,0,0])    # X coord 0 to 1
            c_y = self.get_coefficients(0,15,[0,0,0],[0,0,0])    # Y coord 0 to 0
            c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1
        elif t>=20 and t<35:         # Waypoint 3 to 4 (1,0,1) to (1,1,1) in 15 sec
            c_x = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # X coord 1 to 1
            c_y = self.get_coefficients(0,15,[0,0,0],[1,0,0])    # Y coord 0 to 1
            c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1
        elif t>=35 and t<50:        # Waypoint 4 to 5 (1,1,1) to (0,1,1) in 15 sec
            c_x = self.get_coefficients(0,15,[1,0,0],[0,0,0])    # X coord 1 to 0
            c_y = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Y coord 1 to 1
            c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1
        elif t>=50 and t<=65:       # Waypoint 5 to 6 (0,1,1) to (0,0,1) in 15 sec
            c_x = self.get_coefficients(0,15,[0,0,0],[0,0,0])    # X coord 0 to 0
            c_y = self.get_coefficients(0,15,[1,0,0],[0,0,0])    # Y coord 1 to 0
            c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1
        return (c_x,c_y,c_z)

    def get_position(self,t):
        coeffs = self.traj_eq_coeff(t)
        coeff_x = coeffs[0]
        pos_x = coeff_x[0] + coeff_x[1]*t + coeff_x[2]*(t**2) + coeff_x[3]*(t**3) + coeff_x[4]*(t**4) + coeff_x[5]*(t**5)
        coeff_y = coeffs[1]
        pos_y = coeff_y[0] + coeff_y[1]*t + coeff_y[2]*(t**2) + coeff_y[3]*(t**3) + coeff_y[4]*(t**4) + coeff_y[5]*(t**5)
        coeff_z = coeffs[2]
        pos_z = coeff_z[0] + coeff_z[1]*t + coeff_z[2]*(t**2) + coeff_z[3]*(t**3) + coeff_z[4]*(t**4) + coeff_z[5]*(t**5)
        return (pos_x,pos_y,pos_z)

    def get_velocities(self,t):
        coeffs = self.traj_eq_coeff(t)
        coeff_x = coeffs[0]
        vel_x = coeff_x[1] + 2*coeff_x[2]*t + 3*coeff_x[3]*(t**2) + 4*coeff_x[4]*(t**3) + 5*coeff_x[5]*(t**4)
        coeff_y = coeffs[1]
        vel_y = coeff_y[1] + 2*coeff_y[2]*t + 3*coeff_y[3]*(t**2) + 4*coeff_y[4]*(t**3) + 5*coeff_y[5]*(t**4)
        coeff_z = coeffs[2]
        vel_z = coeff_z[1] + 2*coeff_z[2]*t + 3*coeff_z[3]*(t**2) + 4*coeff_z[4]*(t**3) + 5*coeff_z[5]*(t**4)
        return (vel_x,vel_y,vel_z)
    
    def get_acclerations(self,t):
        coeffs = self.traj_eq_coeff(t)
        coeff_x = coeffs[0]
        acc_x = 2*coeff_x[2] + 6*coeff_x[3]*t + 12*coeff_x[4]*(t**2) + 20*coeff_x[5]*(t**3)
        coeff_y = coeffs[1]
        acc_y = 2*coeff_y[2] + 6*coeff_y[3]*t + 12*coeff_y[4]*(t**2) + 20*coeff_y[5]*(t**3)
        coeff_z = coeffs[2]
        acc_z = 2*coeff_z[2] + 6*coeff_z[3]*t + 12*coeff_z[4]*(t**2) + 20*coeff_z[5]*(t**3)
        return (acc_x,acc_y,acc_z)
    
    def get_full_traj(self):
        # Repeat each coeff calculation for each waypoint and coordinate
        
        # Waypoint 1 to 2 (0,0,0) to (0,0,1) in 5 sec
        c_x = self.get_coefficients(0,5,[0,0,0],[0,0,0])    # X coord 0 to 0
        c_y = self.get_coefficients(0,5,[0,0,0],[0,0,0])    # Y coord 0 to 0
        c_z = self.get_coefficients(0,5,[0,0,0],[1,0,0])    # Z coord 0 to 1

        self.coefficients['p_0']=[c_x,c_y,c_z]

        for t in linspace(0,5):
            x = c_x[0] + c_x[1]*t + c_x[2]*(t**2) + c_x[3]*(t**3) + c_x[4]*(t**4) + c_x[5]*(t**5)
            x_vel = c_x[1] + 2*c_x[2]*(t) + 3*c_x[3]*(t**2) + 4*c_x[4]*(t**3) + 5*c_x[5]*(t**4)
            x_acc = 2*c_x[2] + 6*c_x[3]*(t) + 12*c_x[4]*(t**2) + 20*c_x[5]*(t**3)
            y = c_y[0] + c_y[1]*t + c_y[2]*(t**2) + c_y[3]*(t**3) + c_y[4]*(t**4) + c_y[5]*(t**5)
            y_vel = c_y[1] + 2*c_y[2]*(t) + 3*c_y[3]*(t**2) + 4*c_y[4]*(t**3) + 5*c_y[5]*(t**4)
            y_acc = 2*c_y[2] + 6*c_y[3]*(t) + 12*c_y[4]*(t**2) + 20*c_y[5]*(t**3)
            z = c_z[0] + c_z[1]*t + c_z[2]*(t**2) + c_z[3]*(t**3) + c_z[4]*(t**4) + c_z[5]*(t**5)
            z_vel = c_z[1] + 2*c_z[2]*(t) + 3*c_z[3]*(t**2) + 4*c_z[4]*(t**3) + 5*c_z[5]*(t**4)
            z_acc = 2*c_z[2] + 6*c_z[3]*(t) + 12*c_z[4]*(t**2) + 20*c_z[5]*(t**3)

            self.traj_x.append(x)
            self.traj_y.append(y)
            self.traj_z.append(z)
            self.traj_x_vel.append(x_vel)
            self.traj_y_vel.append(y_vel)
            self.traj_z_vel.append(z_vel)
            self.traj_x_acc.append(x_acc)
            self.traj_y_acc.append(y_acc)
            self.traj_z_acc.append(z_acc)

        # Waypoint 2 to 3 (0,0,1) to (1,0,1) in 15 sec
        c_x = self.get_coefficients(0,15,[0,0,0],[1,0,0])    # X coord 0 to 1
        c_y = self.get_coefficients(0,15,[0,0,0],[0,0,0])    # Y coord 0 to 0
        c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1

        self.coefficients['p_1']=[c_x,c_y,c_z]
        
        for t in linspace(0,15):
            x = c_x[0] + c_x[1]*t + c_x[2]*(t**2) + c_x[3]*(t**3) + c_x[4]*(t**4) + c_x[5]*(t**5)
            x_vel = c_x[1] + 2*c_x[2]*(t) + 3*c_x[3]*(t**2) + 4*c_x[4]*(t**3) + 5*c_x[5]*(t**4)
            x_acc = 2*c_x[2] + 6*c_x[3]*(t) + 12*c_x[4]*(t**2) + 20*c_x[5]*(t**3)
            y = c_y[0] + c_y[1]*t + c_y[2]*(t**2) + c_y[3]*(t**3) + c_y[4]*(t**4) + c_y[5]*(t**5)
            y_vel = c_y[1] + 2*c_y[2]*(t) + 3*c_y[3]*(t**2) + 4*c_y[4]*(t**3) + 5*c_y[5]*(t**4)
            y_acc = 2*c_y[2] + 6*c_y[3]*(t) + 12*c_y[4]*(t**2) + 20*c_y[5]*(t**3)
            z = c_z[0] + c_z[1]*t + c_z[2]*(t**2) + c_z[3]*(t**3) + c_z[4]*(t**4) + c_z[5]*(t**5)
            z_vel = c_z[1] + 2*c_z[2]*(t) + 3*c_z[3]*(t**2) + 4*c_z[4]*(t**3) + 5*c_z[5]*(t**4)
            z_acc = 2*c_z[2] + 6*c_z[3]*(t) + 12*c_z[4]*(t**2) + 20*c_z[5]*(t**3)

            self.traj_x.append(x)
            self.traj_y.append(y)
            self.traj_z.append(z)
            self.traj_x_vel.append(x_vel)
            self.traj_y_vel.append(y_vel)
            self.traj_z_vel.append(z_vel)
            self.traj_x_acc.append(x_acc)
            self.traj_y_acc.append(y_acc)
            self.traj_z_acc.append(z_acc)

        # Waypoint 3 to 4 (1,0,1) to (1,1,1) in 15 sec
        c_x = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # X coord 1 to 1
        c_y = self.get_coefficients(0,15,[0,0,0],[1,0,0])    # Y coord 0 to 1
        c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1

        self.coefficients['p_2']=[c_x,c_y,c_z]
        
        for t in linspace(0,15):
            x = c_x[0] + c_x[1]*t + c_x[2]*(t**2) + c_x[3]*(t**3) + c_x[4]*(t**4) + c_x[5]*(t**5)
            x_vel = c_x[1] + 2*c_x[2]*(t) + 3*c_x[3]*(t**2) + 4*c_x[4]*(t**3) + 5*c_x[5]*(t**4)
            x_acc = 2*c_x[2] + 6*c_x[3]*(t) + 12*c_x[4]*(t**2) + 20*c_x[5]*(t**3)
            y = c_y[0] + c_y[1]*t + c_y[2]*(t**2) + c_y[3]*(t**3) + c_y[4]*(t**4) + c_y[5]*(t**5)
            y_vel = c_y[1] + 2*c_y[2]*(t) + 3*c_y[3]*(t**2) + 4*c_y[4]*(t**3) + 5*c_y[5]*(t**4)
            y_acc = 2*c_y[2] + 6*c_y[3]*(t) + 12*c_y[4]*(t**2) + 20*c_y[5]*(t**3)
            z = c_z[0] + c_z[1]*t + c_z[2]*(t**2) + c_z[3]*(t**3) + c_z[4]*(t**4) + c_z[5]*(t**5)
            z_vel = c_z[1] + 2*c_z[2]*(t) + 3*c_z[3]*(t**2) + 4*c_z[4]*(t**3) + 5*c_z[5]*(t**4)
            z_acc = 2*c_z[2] + 6*c_z[3]*(t) + 12*c_z[4]*(t**2) + 20*c_z[5]*(t**3)

            self.traj_x.append(x)
            self.traj_y.append(y)
            self.traj_z.append(z)
            self.traj_x_vel.append(x_vel)
            self.traj_y_vel.append(y_vel)
            self.traj_z_vel.append(z_vel)
            self.traj_x_acc.append(x_acc)
            self.traj_y_acc.append(y_acc)
            self.traj_z_acc.append(z_acc)

        # Waypoint 4 to 5 (1,1,1) to (0,1,1) in 15 sec
        c_x = self.get_coefficients(0,15,[1,0,0],[0,0,0])    # X coord 1 to 0
        c_y = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Y coord 1 to 1
        c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1

        self.coefficients['p_3']=[c_x,c_y,c_z]
        
        for t in linspace(0,15):
            x = c_x[0] + c_x[1]*t + c_x[2]*(t**2) + c_x[3]*(t**3) + c_x[4]*(t**4) + c_x[5]*(t**5)
            x_vel = c_x[1] + 2*c_x[2]*(t) + 3*c_x[3]*(t**2) + 4*c_x[4]*(t**3) + 5*c_x[5]*(t**4)
            x_acc = 2*c_x[2] + 6*c_x[3]*(t) + 12*c_x[4]*(t**2) + 20*c_x[5]*(t**3)
            y = c_y[0] + c_y[1]*t + c_y[2]*(t**2) + c_y[3]*(t**3) + c_y[4]*(t**4) + c_y[5]*(t**5)
            y_vel = c_y[1] + 2*c_y[2]*(t) + 3*c_y[3]*(t**2) + 4*c_y[4]*(t**3) + 5*c_y[5]*(t**4)
            y_acc = 2*c_y[2] + 6*c_y[3]*(t) + 12*c_y[4]*(t**2) + 20*c_y[5]*(t**3)
            z = c_z[0] + c_z[1]*t + c_z[2]*(t**2) + c_z[3]*(t**3) + c_z[4]*(t**4) + c_z[5]*(t**5)
            z_vel = c_z[1] + 2*c_z[2]*(t) + 3*c_z[3]*(t**2) + 4*c_z[4]*(t**3) + 5*c_z[5]*(t**4)
            z_acc = 2*c_z[2] + 6*c_z[3]*(t) + 12*c_z[4]*(t**2) + 20*c_z[5]*(t**3)

            self.traj_x.append(x)
            self.traj_y.append(y)
            self.traj_z.append(z)
            self.traj_x_vel.append(x_vel)
            self.traj_y_vel.append(y_vel)
            self.traj_z_vel.append(z_vel)
            self.traj_x_acc.append(x_acc)
            self.traj_y_acc.append(y_acc)
            self.traj_z_acc.append(z_acc)
        
        # Waypoint 5 to 6 (0,1,1) to (0,0,1) in 15 sec
        c_x = self.get_coefficients(0,15,[0,0,0],[0,0,0])    # X coord 0 to 0
        c_y = self.get_coefficients(0,15,[1,0,0],[0,0,0])    # Y coord 1 to 0
        c_z = self.get_coefficients(0,15,[1,0,0],[1,0,0])    # Z coord 1 to 1

        self.coefficients['p_4']=[c_x,c_y,c_z]
        
        for t in linspace(0,15):
            x = c_x[0] + c_x[1]*t + c_x[2]*(t**2) + c_x[3]*(t**3) + c_x[4]*(t**4) + c_x[5]*(t**5)
            x_vel = c_x[1] + 2*c_x[2]*(t) + 3*c_x[3]*(t**2) + 4*c_x[4]*(t**3) + 5*c_x[5]*(t**4)
            x_acc = 2*c_x[2] + 6*c_x[3]*(t) + 12*c_x[4]*(t**2) + 20*c_x[5]*(t**3)
            y = c_y[0] + c_y[1]*t + c_y[2]*(t**2) + c_y[3]*(t**3) + c_y[4]*(t**4) + c_y[5]*(t**5)
            y_vel = c_y[1] + 2*c_y[2]*(t) + 3*c_y[3]*(t**2) + 4*c_y[4]*(t**3) + 5*c_y[5]*(t**4)
            y_acc = 2*c_y[2] + 6*c_y[3]*(t) + 12*c_y[4]*(t**2) + 20*c_y[5]*(t**3)
            z = c_z[0] + c_z[1]*t + c_z[2]*(t**2) + c_z[3]*(t**3) + c_z[4]*(t**4) + c_z[5]*(t**5)
            z_vel = c_z[1] + 2*c_z[2]*(t) + 3*c_z[3]*(t**2) + 4*c_z[4]*(t**3) + 5*c_z[5]*(t**4)
            z_acc = 2*c_z[2] + 6*c_z[3]*(t) + 12*c_z[4]*(t**2) + 20*c_z[5]*(t**3)

            self.traj_x.append(x)
            self.traj_y.append(y)
            self.traj_z.append(z)
            self.traj_x_vel.append(x_vel)
            self.traj_y_vel.append(y_vel)
            self.traj_z_vel.append(z_vel)
            self.traj_x_acc.append(x_acc)
            self.traj_y_acc.append(y_acc)
            self.traj_z_acc.append(z_acc)
        
        self.full_trajectory = {'pos':[self.traj_x,self.traj_y,self.traj_z],'vel':[self.traj_x_vel,self.traj_y_vel,self.traj_z_vel],'acc':[self.traj_x_acc,self.traj_y_acc,self.traj_z_acc]}
        return None

    def draw_traj(self):
        t = linspace(0,65)
        self.get_full_traj()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xline = self.traj_x
        yline = self.traj_y
        zline = self.traj_z
        
        ax.plot3D(xline, yline, zline, 'blue')
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title("Desired Trajectory")
        plt.show()

if __name__ == '__main__':
    t=Quintic_Trajectory()
    t.draw_traj()