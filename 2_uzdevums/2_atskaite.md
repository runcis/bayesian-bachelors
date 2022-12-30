# Task 2: Inverse kintetics

Kopumā gāja labi, nebija problemas izveidot atvasinājuma funkcijas un rezultāts sanāca veiksmīgs.

Man ir 2 jautājumi:
1. kāpēc ir nepieciešams loss funkcijas atvasinājums?
2. Kā īstenot roku nepāriešanu pāri otrai? (Es ilgi mēģināju bet nesanāca)


### List of implemented functions

1. Rotation matrix

~~~
def rotation(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ])
~~~

2. Derivative of rotation matrix
~~~
def d_rotation(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([
        [-sin_theta, -cos_theta],
        [cos_theta, -sin_theta],
    ])
~~~

3. Angle rotation algorithm
~~~
d_theta_1 = np.sum(2 * (point_3 - target_point) * (dR1 @ t + dR1 @ R2 @ t))
theta_1 -= d_theta_1 * alpha

d_theta_2 = np.sum(2 * (point_3 - target_point) * (R1 @ dR2 @ t))
theta_2 -= d_theta_2 * alpha

d_theta_3 = np.sum(2 * (point_3 - target_point) * (R2 @ dR3 @ t)  )
theta_3 -= d_theta_3 * alpha
~~~

4. rotacijas aizliegums zem horizonta pirmajai rokai: 
~~~
if (theta_1 < 1.5 and theta_1 > -1.5):
    theta_1 -= d_theta_1 * alpha
else:
    theta_1 += d_theta_1 * alpha
~~~

5. Derivative of mse loss function:
~~~
d_loss = 2*np.mean(target_point - point_3)
~~~

![kinematics arm](../media/kinematics-arm.PNG)