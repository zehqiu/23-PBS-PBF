# The core implementation of PBF algorithm
# reference: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/pbf2d.py

import taichi as ti
from particle import particle_system
from config import *

# @ti.data_oriented
# class pbf():
#     pass
@ti.data_oriented
class pbf:
    def __init__(self, ps):
        self.ps = ps
        self.omegas = ti.Vector.field(dim, float)
        ti.root.dense(ti.i, N_fluid_particles).place(self.omegas)
        
    @ti.kernel
    def prologue(self):
        # 1: for all particles i do
        # 2:    apply forces v_i ⇐ v_i +∆t f_ext(x_i)
        # 3:    predict position x_i^* ⇐ x_i +∆t v_i
        # 4: end for
        # 5: for all particles i do
        # 6:    find neighboring particles Ni(x_i^*)
        # 7: end for
        
        # save old positions to be used in Algorithm 1-21(x_i)
        for i in self.ps.positions:
            self.ps.old_positions[i] = self.ps.positions[i]
        
        # apply gravity within boundary
        for i in self.ps.positions:
            pos, vel = self.ps.positions[i], self.ps.velocities[i]
            vel += gravity * time_delta
            pos += vel * time_delta
            self.ps.positions[i] = self.ps.confine_position_to_boundary(pos)    # check whether hit boundary

        # clear neighbor lookup table
        for I in ti.grouped(self.ps.grid_num_particles):
            self.ps.grid_num_particles[I] = 0
        for I in ti.grouped(self.ps.particle_neighbors):
            self.ps.particle_neighbors[I] = -1

        # update grid
        for p_i in self.ps.positions:
            cell = get_cell(self.ps.positions[p_i])
            # ti.Vector doesn't seem to support unpacking yet
            # but we can directly use int Vectors as indices
            offs = ti.atomic_add(self.ps.grid_num_particles[cell], 1)
            self.ps.grid2particles[cell, offs] = p_i

        # find particle neighbors
        for p_i in self.ps.positions:
            pos_i = self.ps.positions[p_i]
            cell = get_cell(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2),(-1, 2)))):
                cell_to_check = cell + offs
                if is_in_grid(cell_to_check):
                    for j in range(self.ps.grid_num_particles[cell_to_check]):
                        p_j = self.ps.grid2particles[cell_to_check, j]
                        if nb_i < max_num_neighbors and p_j != p_i and (pos_i - self.ps.positions[p_j]).norm() < neighbour_radius:
                            self.ps.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.ps.particle_num_neighbors[p_i] = nb_i
            # if(nb_i<=8):
            #     self.ps.colors[p_i]=ti.Vector([1,1,1])
            # else:
            #     self.ps.colors[p_i]=ti.Vector([0,0,1])

    @ti.kernel
    def epilogue(self):
        # confine to boundary
        for i in self.ps.positions:
            pos = self.ps.positions[i]
            self.ps.positions[i] = self.ps.confine_position_to_boundary(pos)
        # update velocities
        for i in self.ps.positions:
            self.ps.velocities[i] = (self.ps.positions[i] - self.ps.old_positions[i]) / time_delta
        # no vorticity/xsph because we cannot do cross product in 2D...

    @ti.kernel
    def apply_vorticity_confinement(self):
        # compute omega
        for p_i in self.ps.positions:
            pos_i = self.ps.positions[p_i]
            self.omegas[p_i] = pos_i * 0.0
            for j in range(self.ps.particle_num_neighbors[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.ps.positions[p_j]
                grad_j = spiky_gradient(pos_ji, h)
                vij = self.ps.velocities[p_j] - self.ps.velocities[p_i]
                self.omegas[p_i]+= ti.math.cross(vij,grad_j)

        for p_i in self.ps.positions:
            omega_i = self.omegas[p_i]
            if(omega_i.norm()<epsilon):
                continue
            eta = self.ps.positions[p_i] * 0.0
            for j in range(self.ps.particle_num_neighbors[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = self.ps.positions[p_i] - self.ps.positions[p_j]
                grad_j = spiky_gradient(pos_ji, h)
                vij = self.ps.velocities[p_j] - self.ps.velocities[p_i]
                eta+=grad_j * ti.math.length(omega_i)
            if(eta.norm()<epsilon):
                continue
            N = ti.math.normalize(eta)
            f = vorticity_confinement_epsilon * ti.math.cross(N,omega_i)
            self.ps.velocities[p_i]+= f / mass * time_delta
   
    @ti.kernel
    def apply_xsph_viscosity(self):
        for p_i in self.ps.positions:
            x_vesc = self.ps.positions[p_i] * 0.0
            pos_i = self.ps.positions[p_i]
            for j in range(self.ps.particle_num_neighbors[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                vij = self.ps.velocities[p_j] - self.ps.velocities[p_i]
                pos_ji = pos_i - self.ps.positions[p_j]
                vij *= poly6_value(pos_ji.norm(), h)
                x_vesc+= vij
            self.ps.velocities[p_i]+=XSPH_c*x_vesc

    @ti.kernel
    def PBF_solver(self):
        # compute lambdas
        # Eq (8) ~ (11)
        for p_i in self.ps.positions:
            pos_i = self.ps.positions[p_i]

            grad_i = ti.Vector([0.0, 0.0, 0.0])
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.ps.particle_num_neighbors[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.ps.positions[p_j]
                grad_j = spiky_gradient(pos_ji, h)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h)

            # Eq(1)
            density_constraint = (mass * density_constraint / rho0) - 1.0

            sum_gradient_sqr += grad_i.dot(grad_i)
            self.ps.lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)
        
        # compute position deltas
        # Eq(12), (14)
        for p_i in self.ps.positions:
            pos_i = self.ps.positions[p_i]
            lambda_i = self.ps.lambdas[p_i]

            pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
            for j in range(self.ps.particle_num_neighbors[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                lambda_j = self.ps.lambdas[p_j]
                pos_ji = pos_i - self.ps.positions[p_j]
                scorr_ij = compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * spiky_gradient(pos_ji, h)

            pos_delta_i /= rho0
            self.ps.position_deltas[p_i] = pos_delta_i
        
        # apply position deltas
        for i in self.ps.positions:
            self.ps.positions[i] += self.ps.position_deltas[i]

    def run_PBF(self):
        self.prologue()
        for _ in range(pdf_num_iters):
            self.PBF_solver()
        self.epilogue()
        self.apply_vorticity_confinement()
        self.apply_xsph_viscosity()