# deal with particles
import taichi as ti
from config import *
import numpy as np

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
        
        # self.board_states = ti.Vector.field(dim, float)
        
        ti.root.dense(ti.i, N_fluid_particles).place(self.old_positions, self.positions, self.velocities, self.colors)
        grid_snode = ti.root.dense(ti.ijk, grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, N_fluid_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, N_fluid_particles).place(self.lambdas, self.position_deltas)
        # ti.root.place(self.board_states)    
    

    def init_particles(self):
        for i in range(N_fluid_particles):
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
            # self.colors[i] = ti.Vector([0,0,1])
        # self.board_states = ti.Vector([boundary[0] - epsilon, -0.0, 0.0])
    
    # function handling boundary conditions
    @ti.func
    def confine_position_to_boundary(self,p):
        bmin = particle_radius
        # bmax = ti.Vector([self.board_states[0], boundary[1]]) - particle_radius
        bmax = ti.Vector([boundary[0], boundary[1], boundary[2]]) - particle_radius
        for i in ti.static(range(dim)):
            # Use randomness to prevent particles from sticking into each other after clamping
            if p[i] <= bmin:
                p[i] = bmin + epsilon * ti.random()
            elif bmax[i] <= p[i]:
                p[i] = bmax[i] - epsilon * ti.random()
        return p
    