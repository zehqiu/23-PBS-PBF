# The GUI of our project. Containing the input and output setup.
import taichi as ti
from config import *
from particle import particle_system
from pbf import pbf
import os

ti.init(arch=ti.gpu)  # 确定后端

# 初始化数据
width, height = 900, 600
color = (1.0, 1.0, 1.0)

# create particles
ps = particle_system()
print(fluid_blocks_1_z)
ps.init_particles()
# ps.add_rigid_body()
solver = pbf(ps)

# Water tank parameters
tank_vertex = ti.Vector.field(3, dtype=ti.f32, shape=8)
tank_edge = ti.field(dtype=ti.i32, shape=(12, 2))  # 12 edges, each connected to 2 vertices

# create water tank
@ti.kernel
def create_water_tank():
    # left panel
    tank_vertex[0] = [0,0,0]
    tank_vertex[1] = [0,tank_height,0]
    tank_vertex[2] = [0,tank_height,tank_width]
    tank_vertex[3] = [0,0,tank_width]   
    # right panel
    tank_vertex[4] = [tank_depth,0,0]
    tank_vertex[5] = [tank_depth,tank_height,0]
    tank_vertex[6] = [tank_depth,tank_height,tank_width]
    tank_vertex[7] = [tank_depth,0,tank_width]
    # edge
    for i in ti.static(range(4)):
        tank_edge[i, 0] = i
        tank_edge[i, 1] = (i + 1) % 4

        tank_edge[i + 4, 0] = i + 4
        tank_edge[i + 4, 1] = ((i + 1) % 4) + 4

        tank_edge[i + 8, 0] = i
        tank_edge[i + 8, 1] = i + 4

create_water_tank()

# 创建画布
window = ti.ui.Window(name='Position-Based Fluid Simulation', res = (width, height), fps_limit=200, pos = (150, 150))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = ti.ui.Gui(window.get_gui())
camera.position(100, 60, 30)
camera.lookat(30, 20, 30)

# 输入处理的示例(from https://docs.taichi-lang.cn/docs/ggui)
gravity = ti.Vector.field(2, ti.f32, shape=())
attractor_strength = ti.field(ti.f32, shape=())
run_simulate = 0
step_count = 0
output_as_ply = 0
current_directory = os.path.dirname(os.path.realpath(__file__))
movedir = 1
time_period = 0

while window.running:
    # 初始化设置
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    canvas.set_background_color((1, 1, 1))
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(5, 15, 15), color=(1, 1, 1))
    
    # 小窗
    with gui.sub_window("Sub Window", x=0, y=0, width=0.15, height=1):
        gui.text("Some interaction guidance")
        start_simulation = gui.button("Start Simulation")
        export_as_ply = gui.button("Export as PLY")
        reset_scene = gui.button("reset_scene")

        # value = gui.slider_float("name1", value, minimum=0, maximum=100)
        # color = gui.color_edit_3("name2", color)
    
    if start_simulation:
        run_simulate = 1
    if run_simulate == 1:
        step_count = step_count + 1
        if time_period>=0:
            time_period-=1
            ps.move_board(movedir)
        solver.run_PBF()
    if export_as_ply:
        output_as_ply = 1
        count_output = 0
    # 示范用例
    # 用per_vertex_color给颜色数组
    scene.particles(ps.positions, per_vertex_color=ps.colors,radius = particle_radius)
    
    # scene.particles(ps.voxelized_points, color = (0.68, 0.26, 0.19), radius = particle_radius)
    
    # draw water-tank
    scene.lines(tank_vertex, width=3.0, indices=tank_edge, color=(0, 0, 0))
    # scene.rect((0, 0),(ps.board_states[0] / boundary[0], 1),radius=1.5,color=boundary_color,)
    
    # keyboard event processing
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r': pass
        elif window.event.key in [ti.ui.ESCAPE]: break
    # if window.event is not None: gravity[None] = [0, 0]  # if had any event
    # if window.is_pressed(ti.ui.LEFT, 'a'): gravity[None][0] = -1
    # if window.is_pressed(ti.ui.RIGHT, 'd'): gravity[None][0] = 1
    # if window.is_pressed(ti.ui.UP, 'w'): gravity[None][1] = 1
    # if window.is_pressed(ti.ui.DOWN, 's'): gravity[None][1] = -1
    if window.is_pressed('z'):
        ps.add_random_velocities(2)
    if window.is_pressed('c'):
        ps.add_random_velocities(0)
    if window.is_pressed('x'):
        ps.add_random_velocities(1)
    if window.is_pressed(ti.ui.LEFT): 
        movedir = 1
        time_period+=1
    if window.is_pressed(ti.ui.RIGHT): 
        movedir = -1
        time_period+=1
    if window.is_pressed(ti.ui.UP): pass
    if window.is_pressed(ti.ui.DOWN): pass
    # print(gravity)

    # mouse event processing
    mouse = window.get_cursor_pos()
    # ...
    if window.is_pressed(ti.ui.LMB):
        pass      

    if reset_scene:
        ps.init_particles()
        step_count = 0
        run_simulate = 0
        output_as_ply = 0

    if run_simulate==1:
        if output_as_ply == 1:
            if step_count % outputInterval == 0:
                if(count_output==0):
                    os.makedirs(os.path.join(current_directory,"output"), exist_ok=True)

                np_pos = np.reshape(ps.positions.to_numpy(), (N_fluid_particles, 3))
                np_rgb = np.reshape(ps.colors.to_numpy(), (N_fluid_particles, 3))
                # create a PLYWriter
                writer = ti.tools.PLYWriter(num_vertices=N_fluid_particles)
                writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                writer.add_vertex_color(
                    np_rgb[:, 0], np_rgb[:, 1], np_rgb[:, 2])
                writer.export_frame_ascii(count_output, os.path.join(current_directory,series_prefix))
                count_output = count_output + 1
    
    canvas.scene(scene)
    window.show()