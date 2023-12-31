# deal with particles
import taichi as ti
from rigid_config import *
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
        self.n_fluid_particles = 1
        self.n_rigid_particles = 0
        self.n_total_particles = 1
        self.velocities = ti.Vector.field(dim, float)
        self.scale_factor=15.0
        self.displacement_factor=[10,0,10]
        
        self.colors = ti.Vector.field(dim, float)
        
        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        self.lambdas = ti.field(float)

        #fluid
        for i in range(dim):
            self.n_fluid_particles *= len(np.arange(fluid_blocks_1_start[i], fluid_blocks_1_end[i], delta))

        #rigid
        mesh = tm.load(rigid_body_path)
        mesh.vertices *= self.scale_factor
        mesh.vertices += np.array(self.displacement_factor)
        voxelized_mesh = mesh.voxelized(pitch=2*particle_radius).fill()
        voxelized_points_np = voxelized_mesh.points.astype(np.float32)
        self.n_rigid_particles = voxelized_points_np.shape[0]


        # self.voxelized_points = ti.Vector.field(dim, ti.f32, num_particles_obj)
        # self.voxelized_points.from_numpy(voxelized_points_np)


        self.n_total_particles = self.n_fluid_particles + self.n_rigid_particles

        # self.board_states = ti.Vector.field(dim, float)
        ti.root.dense(ti.i, self.n_total_particles).place(self.old_positions, self.positions, self.velocities, self.colors)
        grid_snode = ti.root.dense(ti.ijk, grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, self.n_total_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.n_total_particles).place(self.lambdas, self.position_deltas)
        print(self.n_rigid_particles)
        print(self.n_fluid_particles)
        print(self.n_total_particles)

        # ti.root.place(self.board_states)    
    
    @ti.data_oriented
    def init_particles(self):

        for i in range(self.n_fluid_particles):
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
    
    def add_rigid_body(self, scale_factor=15.0, displacement_factor=[25,0,25]):
        mesh = tm.load(rigid_body_path)
        mesh.vertices *= scale_factor
        mesh.vertices += np.array(displacement_factor)
        voxelized_mesh = mesh.voxelized(pitch=2*particle_radius).fill()
        voxelized_points_np = voxelized_mesh.points.astype(np.float32)
        print(voxelized_points_np)
        for i in range(self.n_rigid_particles):
            posx = mesh.vertices[i][0]
            posy = mesh.vertices[i][1]
            posz = mesh.vertices[i][2]
            self.positions[i + self.n_fluid_particles] = ti.Vector([posx, posy, posz])

        # voxelized_mesh = mesh.voxelized(pitch=2*particle_radius).fill()
        # voxelized_points_np = voxelized_mesh.points.astype(np.float32)
        # # voxelized_points_np *= scale_factor  # Scale the positions
        # num_particles_obj = voxelized_points_np.shape[0]
        
        # self.voxelized_points = ti.Vector.field(dim, ti.f32, num_particles_obj)
        # self.voxelized_points.from_numpy(voxelized_points_np)
        

    # function handling boundary conditions
    @ti.func
    def confine_position_to_boundary(self, p):
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

    