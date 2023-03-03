from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from pathlib import Path

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler('params.txt'))

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

def boundary(x, on_boundary):
	return on_boundary

for i in range(10):
	PATH = Path(f'poisson_{i}/')
	os.makedirs(PATH, exist_ok=True)

	# Define boundary condition
	# u_e = a_0 + a_1*x + a_2*y + a_3*x*y + a_4*x**2 + a_5*y**2
	params = np.random.randn(6)
	logger.info(params)
	u_D = Expression('a_0 + a_1*x[0] + a_2*x[1] + a_3*x[0]*x[1] + a_4*x[0]*x[0] + a_5*x[1]*x[1]', degree=2,
		              a_0=params[0], a_1=params[1], a_2=params[2], a_3=params[3], a_4=params[4], a_5=params[5])

	bc = DirichletBC(V, u_D, boundary)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	a = dot(grad(u), grad(v))*dx
	f = Constant(-2*(params[4] + params[5]))
	L = f*v*dx

	# Compute solution
	u = Function(V)
	solve(a == L, u, bc)

	# Save solution to file in VTK format
	vtkfile = File(str(PATH / 'solution.pvd'))
	vtkfile << u
