# deal with particles
import taichi as ti
from config import *
import numpy as np
import trimesh as tm

@ti.data_oriented
class Material:
    Fluid = 0
    Rigid = 1
    Foam = 3

@ti.data_oriented
class particle_system():    
    def __init__(self) -> None:
        # init fluid particles
        self.positions = ti.Vector.field(dim, float)
        self.old_positions = ti.Vector.field(dim, float)
        self.position_deltas = ti.Vector.field(dim, float)
        
        self.velocities = ti.Vector.field(dim, float)
        
        self.colors = ti.Vector.field(dim, float)
        
        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        self.lambdas = ti.field(float)
        
        # 0- x value, 1- board velocity, 2 - time
        # self.board_states = ti.Vector.field(3,float)
        self.board_states = ti.field(float)
        
        ti.root.dense(ti.i, N_fluid_particles).place(self.old_positions, self.positions, self.velocities, self.colors)
        grid_snode = ti.root.dense(ti.ijk, grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, N_fluid_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, N_fluid_particles).place(self.lambdas, self.position_deltas)
        ti.root.place(self.board_states)    
    

    def init_particles(self):
        for i in range(N_fluid_1_particles):
            y = i // (fluid_blocks_1_x * fluid_blocks_1_z)
            z = (i // fluid_blocks_1_x) % fluid_blocks_1_z
            x = i % (fluid_blocks_1_x)
            posx = fluid_blocks_1_start[0] + x * delta
            posy = fluid_blocks_1_start[1] + y * delta
            posz = fluid_blocks_1_start[2] + z * delta
            self.positions[i] = ti.Vector([posx, posy,posz])

            # self.positions[i] = ti.Vector([i % fluid_blocks_1_end[0],
            #                                i % (fluid_blocks_1_end[0] * fluid_blocks_1_end[2]),
            #                                i // (fluid_blocks_1_end[0] * fluid_blocks_1_end[2])]) * delta
            # for c in ti.static(range(dim)):
            # self.velocities[i] = ti.Vector([0, -10, 0])
            self.colors[i] = color1
        for i in range(0, N_fluid_2_particles):
            y = i // (fluid_blocks_2_x * fluid_blocks_2_z)
            z = (i // fluid_blocks_2_x) % fluid_blocks_2_z
            x = i % (fluid_blocks_2_x)
            posx = fluid_blocks_2_start[0] + x * delta
            posy = fluid_blocks_2_start[1] + y * delta
            posz = fluid_blocks_2_start[2] + z * delta
            self.positions[i+N_fluid_1_particles] = ti.Vector([posx, posy,posz])

            # self.positions[i] = ti.Vector([i % fluid_blocks_1_end[0],
            #                                i % (fluid_blocks_1_end[0] * fluid_blocks_1_end[2]),
            #                                i // (fluid_blocks_1_end[0] * fluid_blocks_1_end[2])]) * delta
            # for c in ti.static(range(dim)):
            # self.velocities[i] = ti.Vector([0, -10, 0])
            self.colors[i+N_fluid_1_particles] = color2
        # self.board_states = ti.Vector([0, 1, 0.0])
        self.board_states[None] = 0.0
    
    def add_rigid_body(self,scale_factor=15.0,displacement_factor=[10,0,10]):
        mesh = tm.load(rigid_body_path)
        mesh.vertices *= scale_factor
        mesh.vertices += np.array(displacement_factor)
        voxelized_mesh = mesh.voxelized(pitch=2*particle_radius).fill()
        voxelized_points_np = voxelized_mesh.points.astype(np.float32)
        # voxelized_points_np *= scale_factor  # Scale the positions
        num_particles_obj = voxelized_points_np.shape[0]
        self.voxelized_points = ti.Vector.field(dim, ti.f32, num_particles_obj)
        self.voxelized_points.from_numpy(voxelized_points_np)

    # function handling boundary conditions
    @ti.func
    def confine_position_to_boundary(self,p):
        bmin = ti.Vector([0,0,self.board_states[None]+epsilon])+particle_radius
        # bmax = ti.Vector([self.board_states[0], boundary[1]]) - particle_radius
        bmax = ti.Vector([boundary[0], boundary[1], boundary[2]]) - particle_radius
        for i in ti.static(range(dim)):
            if p[i] <= bmin[i]:
                p[i] = bmin[i] + epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - epsilon * ti.random()
        return p
    
    @ti.kernel
    # Interaction: Moving Board
    def move_board(self,flag:int):
        # probably more accurate to exert force on particles according to hooke's law.
         # 0- x value, 1- board velocity, 2 - time
        # self.board_states[1] = flag
        vel_strength = 3
        if 0 < self.board_states[None] + flag * time_delta * vel_strength < boundary[2] / 3:
            self.board_states[None] = self.board_states[None] + flag * time_delta * vel_strength
        print("flag:", flag)
        print("board_states:", self.board_states[None])
        
    @ti.kernel
    def add_random_velocities(self,k:int):
        for i in range(N_fluid_particles):
            self.velocities[i][k] += ti.random()
    