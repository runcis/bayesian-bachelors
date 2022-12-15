import numpy as np
import sys
import math
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 7) # size of window
plt.ion()
plt.style.use('dark_background')

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])

is_running = True
def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app
        sys.exit(0)

def on_close(event):
    global is_running
    is_running = False

def rotation(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ])

def d_rotation(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([
        [-sin_theta, -cos_theta],
        [cos_theta, -sin_theta],
    ])

fig, _ = plt.subplots()
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(0)
theta_2 = np.deg2rad(0)
theta_3 = np.deg2rad(0)
alpha = 1e-2


while is_running:
    plt.clf()
    
    t = np.array([0.0, 1.0]) * length_joint

    R1 = rotation(theta_1)
    dR1 = d_rotation(theta_1)
    R2 = rotation(theta_2)
    dR2 = d_rotation(theta_2)
    R3 = rotation(theta_3)
    dR3 = d_rotation(theta_3)

    point_1 = np.dot(R1,t)
    point_2 = point_1 + np.dot(R1, np.dot(R2, t))
    point_3 = point_2 + np.dot(R2, np.dot(R3, t))
   
    joints = []
    joints.append(anchor_point)
    joints.append(point_1)
    joints.append(point_2)
    joints.append(point_3)

    np_joints = np.array(joints)

    d_theta_1 = np.sum(2 * (point_3 - target_point) * (dR1 @ t + dR1 @ R2 @ t))
    
    # check if didn't go below
    # TODO change this doesn't seem right
    if (theta_1 < 1.5 and theta_1 > -1.5):
        theta_1 -= d_theta_1 * alpha
    else:
        theta_1 += d_theta_1 * alpha

    d_theta_2 = np.sum(2 * (point_3 - target_point) * (R1 @ dR2 @ t))
    theta_2 -= d_theta_2 * alpha

    d_theta_3 = np.sum(2 * (point_3 - target_point) * (R2 @ dR3 @ t)  )
    theta_3 -= d_theta_3 * alpha
        
    loss = np.mean((target_point - point_3) **2)
    d_loss = 2*np.mean(target_point - point_3)


    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')
    
    plt.title(f'd_loss = {d_loss}')
    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-3)
    