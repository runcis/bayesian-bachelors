import numpy as np
import matplotlib.pyplot as plt
import math

is_running = True

degrees1 = -10
degrees2 = -10

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])
length_joint = 2.0

def rotation(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ])

def press(event):
    global is_running,degrees1,degrees2
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app

    elif event.key == 'right':
        degrees1 += 1
        print(degrees1)

    elif event.key == 'left':
        degrees1 -= 1
  
    elif event.key == 'up':
        degrees2 += 1

    elif event.key == 'down':
        degrees2 -= 1


def on_close(event):
    global is_running
    is_running = False

def distanceBetweenTwoPoints(pos1, pos2):
    return np.sum((pos1 - pos2)**2)/2

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('close_event', on_close)

while is_running:

    theta_1 = np.deg2rad(degrees1)
    theta_2 = np.deg2rad(degrees2)

    plt.clf()


    t = np.array([0.0, 1.0]) * length_joint

    R1 = rotation(theta_1)
    R2 = rotation(theta_2)

    point_1 = np.dot(R1,t)
    point_2 = point_1 + np.dot(R1, np.dot(R2, t))
   
    joints = []
    joints.append(anchor_point)
    joints.append(point_1)
    joints.append(point_2)

    np_joints = np.array(joints)

    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    distance = distanceBetweenTwoPoints(target_point, point_2)

    plt.title(f'theta_1: {round(np.rad2deg(theta_1))} theta_2: {round(np.rad2deg(theta_2))} distance: {distance}')

    plt.xlim(-5, 5)
    plt.ylim(0, 10)
    plt.draw()
    plt.pause(1e-1)
    #y = R(theta1)*t + R(theta2)*(R(theta2)*t)

input('end')
