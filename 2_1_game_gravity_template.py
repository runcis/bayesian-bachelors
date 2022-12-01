import numpy as np

import matplotlib
import time
import math
matplotlib.use("TkAgg")


### UI ATTRIBUTES ###

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 7) # size of window
plt.ion()
plt.style.use('dark_background')

SPACE_SIZE = 10
is_running = True

### VECTOR AND MATRIX TRANSFORMATIONS ###

def rotation_mat(degrees):
    rad = np.radians(degrees)
    cos_x = math.cos(rad)
    sin_x = math.sin(rad)
    R = np.array([
        [cos_x, -sin_x, 0.0],
        [sin_x, cos_x, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return R

def translation_mat(dx, dy):
    T = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0]
    ])
    return T

def scale_mat(dx, dy):
    T = np.array([
        [dx, 0.0, 0.0],
        [0.0, dy, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return T
    
def clearMatrix(matrix):
    nullMatrix = np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    return dot(matrix,nullMatrix)

def dot(X, Y):
    is_transposed = False

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[1] != Y.shape[0]:
        is_transposed = True
        Y = np.transpose(Y)
    
    X_rows = X.shape[0]
    Y_columns = Y.shape[1]

    for X_row in range(X_rows):
        for Y_column in range(Y_columns):
            product[X_row, Y_column] = np.sum(X[X_row,:] * Y[:, Y_column])

    if is_transposed:
        product = np.transpose(product)
    
    if product.shape[0] == 1:
        product = product.flatten()

    return product


def vec2d_to_vec3d(vec2d):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    b = np.array([0, 0, 1])
    res3d = dot(I, vec2d) + b
    return res3d


def vec3d_to_vec2d(vec3d):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    res2d = dot(I, vec3d)
    return res2d

def l2_normalize_vec2d(vec2d):
    length = math.sqrt(vec2d[0]**2 + vec2d[1]**2)
    normalized_vec2d = np.array([vec2d[0]/length, vec2d[1]/length])
    return normalized_vec2d

### OBJECTS ###

class MovableObject(object):
    def __init__(self):
        super().__init__()
        self.__angle = np.random.random() * np.pi

        self.geometry = []
        self.attribute_name = 'Noname'

        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)
        self.T_init = np.identity(3)


        self.vec_pos = np.zeros((2,))
        self.vec_dir_init = np.array([0.0, 1.0])
        self.vec_dir = np.copy(self.vec_dir_init)

        self.external_forces = np.zeros((5, 2))
        self.speed = 0.8

    def set_angle(self, angle):
        self.__angle = angle
        self.R = rotation_mat(angle)

        vec3d = vec2d_to_vec3d(self.vec_dir_init)
        vec3d = dot(self.R, vec3d)
        self.vec_dir = vec3d_to_vec2d(vec3d)

        self.__update_transformation()

    def get_angle(self):
        return self.__angle

    def update_movement(self, dt):

        # handle borders (fly out other side)
        if self.vec_pos[1] > 10:
            self.vec_pos[1] = -10

        if self.vec_pos[1] < -10:
            self.vec_pos[1] = 10

        if self.vec_pos[0] > 10:
            self.vec_pos[0] = -10

        if self.vec_pos[0] < -10:
            self.vec_pos[0] = 10


        self.vec_pos += self.vec_dir * self.speed * dt
        self.vec_pos += np.mean(self.external_forces * dt, axis=0)
        self.__update_transformation()


    def __update_transformation(self):
        self.T = translation_mat(self.vec_pos[0], self.vec_pos[1])
        
        self.C = np.identity(3)
        self.C = dot(self.C, self.T)
        self.C = dot(self.C, self.R)
        self.C = dot(self.C, self.S)
        self.C = dot(self.C, self.T_init)

    def draw(self):
        x_values = []
        y_values = []
        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)
            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)

            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        plt.plot(x_values, y_values)

class Player(MovableObject):
    def __init__(self):
        super().__init__()
        self.attribute_name = 'Player'

        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])

        self.vec_pos = np.array([0.0, 0.0])
        self.speed = 0.75

        self.T_init = translation_mat(0 , -0.5)
        self.S = scale_mat(0.2, 0.8)


    def activate_thrusters(self):
        self.speed += 1.3

        pass

    def update_movement(self, dt):
        self.speed -= dt * 0.75
        self.speed = max(0.75, self.speed)
        super().update_movement(dt)

def drawCircle(radius):
        detail = 24
        circle = []
        d = 0
        x = 0
        while d < 375:
            circle.append([radius*np.cos(np.radians(d)), radius*np.sin(np.radians(d))])
            d +=375/detail
            x +=1

        return np.array(circle)

class Planet(MovableObject):
    def __init__(self, name, index, radius):
        super().__init__()
        self.attribute_name = name
        self.speed = 0
        self.planetNumber = index
        print(radius)
        self.radius = radius

        s = drawCircle(self.radius)
        self.geometry = s

        self.vec_pos = np.array([np.random.uniform(-10.0, 10.0), np.random.uniform(-10.0, 10.0)])
        self.speed = 0
    
    def update_movement(self, dt):
        if isCollided(self, player):
            closeWithGameOver()
            
        super().update_movement(dt)

        updateForceOnPlayer(self)

class EmissionParticle(MovableObject):
    def __init__(self, directionVector, position):
        super().__init__()
        self.speed = .75
        I = np.array([
            [1, 0],
            [0, 1],
        ])

        self.vec_pos = dot(position, I)

        radius = np.random.uniform(0.15, 0.3)

        s = drawCircle(radius)
        self.geometry = s

        directionChangeMatrix = np.array([
            [np.random.uniform(-1.5, -0.5), 0],
            [0, np.random.uniform(-1.5, -0.5)],
        ])
        self.vec_dir = dot(directionVector, directionChangeMatrix)
        self.lifespan = 1
    def update_movement(self, dt):
        self.lifespan -= dt
        super().update_movement(dt)
        self.geometry = self.geometry * .75
        self.speed -= dt * 0.6
        if self.lifespan < 0:
            self.geometry = clearMatrix(self.geometry)

### METHODS ###

def isCollided(firstObject:MovableObject, secondObject:MovableObject):
    d_2 = distanceBetweenTwoObjects(firstObject.vec_pos, secondObject.vec_pos)
    if d_2 < 0.2:
        return True
    return False

def distanceBetweenTwoObjects(pos1, pos2):
    return np.sum((pos1 - pos2)**2)/2

def closeWithGameOver():
    plt.text(x=-SPACE_SIZE+9, y=SPACE_SIZE-9, s=f'GAME OVER')
    plt.pause(5)
    global is_running
    is_running = False

def updateForceOnPlayer(self:MovableObject):
    F = 9.82 * self.radius / distanceBetweenTwoObjects(self.vec_pos, player.vec_pos)*2
    F_vec = l2_normalize_vec2d(self.vec_pos - player.vec_pos)
    player.external_forces[self.planetNumber] = F * F_vec


def createEmissionParticles(player):
    particles = []
    particles.append(EmissionParticle(player.vec_dir, player.vec_pos))
    particles.append(EmissionParticle(player.vec_dir, player.vec_pos))
    particles.append(EmissionParticle(player.vec_dir, player.vec_pos))
    return np.array(particles)

def press(event):
    print('press', event.key)
    if event.key == 'escape':
        global is_running
        is_running = False # quits app

    elif event.key == 'right':
        player.set_angle(player.get_angle()-5)

    elif event.key == 'left':
        player.set_angle(player.get_angle()+5)

    elif event.key == ' ':
        player.activate_thrusters()
        actors.extend(createEmissionParticles(player))
            

def on_close(event):
    global is_running
    is_running = False

### GAMESPACE INITIALIZATIONS ###

player: Player = Player()
zeme: Planet = Planet("zeme", 0, 1)
marss: Planet = Planet("marss", 1, 0.5)
jupiters: Planet = Planet("jupiters", 2, 2)
actors = [player, zeme, marss, jupiters]

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('close_event', on_close)
dt = 1e-4

last_time = time.time()
zero_time = time.time()

### GAME LOOP ###

while is_running:
    plt.clf()
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.text(x=-SPACE_SIZE+0.1, y=SPACE_SIZE-0.4, s=f'angle: {player.get_angle()}')

    plt.xlim(-SPACE_SIZE, SPACE_SIZE)
    plt.ylim(-SPACE_SIZE, SPACE_SIZE)


    real_dt = time.time() - last_time
    for actor in actors: # polymorhism
        actor.update_movement(real_dt)
    last_time = time.time()

    time_passed = last_time - zero_time
    if time_passed > 5 and time_passed < 5.05 :
        actors.append(Planet("new", 3, np.random.uniform(0.5, 2.5)))
        
    if time_passed > 15 and time_passed < 15.05 :
        actors.append(Planet("new", 4, np.random.uniform(0.5, 2.5)))

    for actor in actors: # polymorhism
        actor.draw()
    
    plt.draw()
    plt.pause(dt)

print('exit')