import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_delta_x(mass, charge):
    h_bar = 1.05457e-34  # Planck's constant divided by 2*pi
    delta_p = h_bar / (2 * math.pi)
    delta_x = h_bar / (2 * math.pi * delta_p)
    return delta_x

def solve_time_independent_schrodinger(mass, charge, num_points, box_size):
    h_bar = 1.05457e-34  # Planck's constant divided by 2*pi
    dx = box_size / num_points
    x = np.linspace(-box_size/2, box_size/2, num_points)
    psi = np.zeros((num_points,), dtype=np.complex128)
    psi[int(num_points/2)] = 1.0

    V = np.zeros((num_points,), dtype=np.float64)  # Potential energy
    # Set up your potential energy function V(x) here

    kinetic_energy = -h_bar**2 / (2 * mass * dx**2)
    diagonal = kinetic_energy + V
    off_diagonal = -kinetic_energy / 2
    H = np.diag(diagonal) + np.diag(off_diagonal*np.ones(num_points-1), k=-1) + np.diag(off_diagonal*np.ones(num_points-1), k=1)

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    psi_t = np.conj(eigenvectors.T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, np.arange(psi_t.shape[1]))
    Z = np.abs(psi_t[:, :psi_t.shape[1]])**2
    ax.plot_surface(X, Y, Z, cmap='plasma')  # Updated color to 'plasma'
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Energy Level')
    ax.set_zlabel('Probability Density')
    ax.set_title('Time-Independent Schrödinger Equation')
    plt.show()

    return np.abs(psi_t)**2

def solve_time_dependent_schrodinger(mass, charge, num_points, box_size, total_time, time_step):
    h_bar = 1.05457e-34  # Planck's constant divided by 2*pi
    dx = box_size / num_points
    dt = time_step

    x = np.linspace(-box_size/2, box_size/2, num_points)
    psi = np.zeros((num_points,), dtype=np.complex128)
    psi[int(num_points/2)] = 1.0

    V = np.zeros((num_points,), dtype=np.float64)
    # Set up your potential energy function V(x) here

    num_steps = int(total_time / dt)
    for step in range(num_steps):
        psi_new = np.zeros_like(psi)
        for i in range(num_points):
            lap_psi = (psi[i-1] - 2 * psi[i] + psi[(i+1) % num_points]) / dx**2
            H = (h_bar**2 / (2 * mass)) * lap_psi + V[i] * psi[i]
            psi_new[i] = psi[i] - (1j * H * dt / h_bar)
        psi = psi_new

    quantum_map = np.abs(psi)**2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, np.arange(num_steps))
    Z = np.abs(psi[np.newaxis, :]) ** 2
    ax.plot_surface(X, T, Z, cmap='plasma')  # Updated color to 'plasma'
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Time')
    ax.set_zlabel('Probability Density')
    ax.set_title('Time-Dependent Schrödinger Equation')
    plt.show()

    return quantum_map

mass = float(input("Enter the mass of the particle: "))
charge = float(input("Enter the charge of the particle: "))

delta_x = calculate_delta_x(mass, charge)
print("Delta x:", delta_x)

num_points = 100  # Number of points in one dimension
box_size = 1e-8  # Size of the simulation box in meters

quantum_map = solve_time_independent_schrodinger(mass, charge, num_points, box_size)
for i in (quantum_map):
    print("Value")
    print(i)
total_time = 1e-15  # Total simulation time in seconds
time_step = 1e-18  # Time step in seconds

quantum_map = solve_time_dependent_schrodinger(mass, charge, num_points, box_size, total_time, time_step)
for i in (quantum_map):
   print(i)
