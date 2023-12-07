# The GUI of our project. Containing the input and output setup.
import taichi as ti
from config import *
from particle import particle_system
from pbf import pbf

ti.init(arch=ti.gpu)  # 确定后端

# 初始化数据
width, height = 900, 600
color = (1.0, 1.0, 1.0)

# create particles
ps = particle_system()
ps.init_particles()
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

while window.running:    
    # 初始化设置
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    canvas.set_background_color((1, 1, 1))
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    
    # 小窗
    with gui.sub_window("Sub Window", x=0, y=0, width=0.15, height=1):
        gui.text("Some interaction guidance")
        select_point_mode = gui.button("Select Particles")
        explosion_mode = gui.button("Explosion")

        # value = gui.slider_float("name1", value, minimum=0, maximum=100)
        # color = gui.color_edit_3("name2", color)
    
    if select_point_mode:
        run_simulate = 1
    if run_simulate == 1:
        solver.run_PBF()
    # 示范用例
    scene.particles(ps.positions, radius = particle_radius)
    
    # draw water-tank
    scene.lines(tank_vertex, width=3.0, indices=tank_edge, color=(0, 0, 0))
    
    # keyboard event processing
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r': pass
        elif window.event.key in [ti.ui.ESCAPE]: break
    # if window.event is not None: gravity[None] = [0, 0]  # if had any event
    # if window.is_pressed(ti.ui.LEFT, 'a'): gravity[None][0] = -1
    # if window.is_pressed(ti.ui.RIGHT, 'd'): gravity[None][0] = 1
    # if window.is_pressed(ti.ui.UP, 'w'): gravity[None][1] = 1
    # if window.is_pressed(ti.ui.DOWN, 's'): gravity[None][1] = -1
    if window.is_pressed(ti.ui.LEFT): pass
    if window.is_pressed(ti.ui.RIGHT): pass
    if window.is_pressed(ti.ui.UP): pass
    if window.is_pressed(ti.ui.DOWN): pass
    # print(gravity)

    # mouse event processing
    mouse = window.get_cursor_pos()
    # ...
    if window.is_pressed(ti.ui.LMB):
        pass      
        
    canvas.scene(scene)
    window.show()