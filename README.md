This is the source code for our implementation of position based fluid. It's a visual demonstration on how the particles (a representative of 3d field information) move and interact.

The program highly relies on Taichi framework. The following packages are required to successfully run the code.
- Taichi (= 1.6.0)
- Numpy
- Trimesh (= 4.0.3)

We had some issue loading configure files to Taichi, therefore we divide the two showcases of multi fluid blocks and fluid-rigid coupling into two separate function blocks. You can check each out
by running main.py and rigid_main.py.
