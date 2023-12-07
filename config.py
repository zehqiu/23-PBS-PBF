# this file includes some initial state of the structure.
import math
import taichi as ti
# Taichi parameter
device = 'gpu'

# The dimension of the project
dim = 3
# gravity
gravity = ti.Vector([0,-9.8,0])
# time step slot
time_delta = 1.0 / 20.0
# epsilon in equation
epsilon = 1e-5

# water tank parameters
tank_width, tank_height, tank_depth = 50, 30, 30
# boundary
boundary = (tank_width, tank_height, tank_depth)

cell_size = 2.5
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

# Gui parameters
background_color = (1.0, 1.0, 1.0)

# particle parameters
# The radius of particle
particle_radius = 0.1
# The number of fluid particles
fluid_blocks = 1
fluid_block_loc = (10,20,10)
N_fluid_block_x = 10
N_fluid_block_y = 30
N_fluid_block_z = 30
N_fluid_particles = N_fluid_block_x * N_fluid_block_y * N_fluid_block_z
# pre-defined parameter
max_num_particles_per_cell = 100
max_num_neighbors = 100


# PBF parameters
# kernel radius
h = 1.1
# neighbour radius
neighbour_radius = h * 1.05
# fluid mass
mass = 1.0
# fluid density
rho0 = 1.0
# epsilon in equation
lambda_epsilon = 100.0
# sub_step number
pdf_num_iters = 5
# correction parameter
corr_deltaQ_coeff = 0.3
corrK = 0.001
corrN = 4.0

# kernel function coeff
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1] and c[2] >= 0 and c[2] < grid_size[2] 

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    # pow(x, 4)
    x = ti.pow(x,4)
    return (corrK) * x