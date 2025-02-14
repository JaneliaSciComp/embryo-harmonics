# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Poisson problem with pure Neumann boundary conditions
# The Poisson problem with pure Neumann boundary conditions
# $\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx$ with $\nabla u \cdot \vec{n} = 0$ on $\partial \Omega$ has no unique solution.
# In fact, it's easy to see that for every solution $u \in H^1(\Omega)$, the function $u + c$ with $c \in \mathbb{R}$ is also a solution.
#
# The usual approach is to factor out the kernel of the differential operator by requiring that $\int_\Omega u \, dx = 0$.
#
# Altogether, this leads to the following weak saddle-point problem. Find $u \in V := H^1(\Omega)$ and $p \in Q := \mathbb{R}$ such that for all $(v, q) \in V \times Q$ it holds that
#
# $\int_\Omega \nabla u \cdot \nabla v \, dx + p \int_\Omega v \, dx = \int_\Omega f v \, dx$
#
# $q \int_\Omega u \, dx = 0\, dx$

# %%
from ngsolve import *
from netgen.geom2d import unit_square
from ngsolve.webgui import Draw

# %%
# Set up a triangulation of the unit square and the necessary finite element spaces
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
V = H1(mesh, order=1)
Q = FESpace("number", mesh)
fes = V * Q

# %%
# Formulate saddle-point problem
(u, p), (v, q) = fes.TnT()

a = BilinearForm(fes)
a += grad(u) * grad(v) * dx
a += p * v * dx
a += q * u * dx

f = LinearForm(fes)
f += x * y * v * dx  # f(x, y) = xy

a.Assemble()
f.Assemble()

# %%
# Solve the whole system for the function and the Lagrange multiplier
solution = GridFunction(fes)
solution.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# %%
# Visualize the function part of the solution
print(f"Integral of u: {Integrate(solution.components[0], mesh)}")
print(f"Value of Lagrange multiplier: {solution.components[1].vec}")
Draw(solution.components[0])
