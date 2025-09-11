import sympy as sp

# Time variable
t = sp.symbols('t')

# State variables as functions of time
x = sp.Function('x')(t)
theta = sp.Function('theta')(t)

# Parameters
m, M, l, g = sp.symbols('m M l g')
F = sp.symbols('F')  # external force on the cart

# Velocities
x_dot = sp.diff(x, t)
theta_dot = sp.diff(theta, t)

# Kinetic energy
cart_kin = 0.5 * M * x_dot**2
pend_kin = 0.5 * m * (x_dot**2 + (l*theta_dot)**2 + 2*x_dot*l*theta_dot*sp.cos(theta))

# Potential energy
pot = m * g * l * sp.cos(theta)

# Lagrangian
L = cart_kin + pend_kin - pot

# Generalized coordinates and velocities
q = [x, theta]
q_dot = [x_dot, theta_dot]

# Euler-Lagrange expressions
F_expr = sp.diff(sp.diff(L, x_dot), t) - sp.diff(L, x)
tau_expr = sp.diff(sp.diff(L, theta_dot), t) - sp.diff(L, theta)

# Define symbols for accelerations
x_ddot, theta_ddot = sp.symbols('x_ddot theta_ddot')

# Replace second derivatives in Euler-Lagrange expressions
subs_dict = {sp.diff(x, t, 2): x_ddot, sp.diff(theta, t, 2): theta_ddot}
F_eq = F_expr.subs(subs_dict) - F        # include input force
tau_eq = tau_expr.subs(subs_dict)        # pendulum torque = 0 (no external torque)

# Solve the nonlinear system for x_ddot and theta_ddot
solution = sp.solve([F_eq, tau_eq], (x_ddot, theta_ddot), simplify=True)

# Display the nonlinear accelerations
print("Nonlinear x_ddot:")
sp.pprint(solution[x_ddot])

print("\nNonlinear theta_ddot:")
sp.pprint(solution[theta_ddot])