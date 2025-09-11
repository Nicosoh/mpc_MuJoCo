import sys

import casadi
import hppfcl as fcl
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

def make_cartpole(ub=True): # ub: unbounded
    model = pin.Model()

    m1 = 1.0
    m2 = 0.1
    length = 0.5
    base_sizes = (0.4, 0.2, 0.05)

    base = pin.JointModelPX() # PX: prismatic joint along x-axis
    base_id = model.addJoint(0, base, pin.SE3.Identity(), "base")

    if ub:
        pole = pin.JointModelRUBY() # RUBY: revolute unbounded along y-axis
    else:
        pole = pin.JointModelRY() # RY: revolute along y-axis
    pole_id = model.addJoint(1, pole, pin.SE3.Identity(), "pole")

    base_inertia = pin.Inertia.FromBox(m1, *base_sizes) # mass, lx, ly, lz
    pole_inertia = pin.Inertia( # mass, location of center of mass, inertia matrix
        m2,
        np.array([0.0, 0.0, length / 2]),
        m2 / 5 * np.diagflat([1e-2, length**2, 1e-2]),
    )

    base_body_pl = pin.SE3.Identity() # base body placement (translation and rotation)
    pole_body_pl = pin.SE3.Identity() # pole body placement (translation and rotation)
    pole_body_pl.rotation = pin.utils.rpyToMatrix(np.array([0.0, np.pi/2, 0.0]))
    pole_body_pl.translation = np.array([0.0, 0.0, length / 2]) # move pole up by length/2

    model.appendBodyToJoint(base_id, base_inertia, base_body_pl) # attach body to joint
    model.appendBodyToJoint(pole_id, pole_inertia, pole_body_pl) # attach body to joint

    # make visual/collision models (not used in dynamics)
    collision_model = pin.GeometryModel()
    shape_base = fcl.Box(*base_sizes)
    radius = 0.01
    shape_pole = fcl.Capsule(radius, length)
    RED_COLOR = np.array([1, 0.0, 0.0, 1.0])
    WHITE_COLOR = np.array([1, 1.0, 1.0, 1.0])
    geom_base = pin.GeometryObject("link_base", base_id, shape_base, base_body_pl)
    geom_base.meshColor = WHITE_COLOR
    geom_pole = pin.GeometryObject("link_pole", pole_id, shape_pole, pole_body_pl)
    geom_pole.meshColor = RED_COLOR

    collision_model.addGeometryObject(geom_base)
    collision_model.addGeometryObject(geom_pole)
    visual_model = collision_model
    return model, collision_model, visual_model


class PinocchioCasadi:
    """Take a Pinocchio model, turn it into a Casadi model
    and define the appropriate graphs.
    """

    def __init__(self, model: pin.Model, timestep=0.05):
        self.model = model
        self.cmodel = cpin.Model(model)  # cast to CasADi model
        self.cdata = self.cmodel.createData() # create CasADi data
        self.timestep = timestep
        self.create_dynamics()
        self.create_discrete_dynamics()

    def create_dynamics(self):
        """Create the acceleration expression and acceleration function."""
        nq = self.model.nq              # number of configuration variables
        nu = 1                          # number of control inputs
        nv = self.model.nv              # number of velocity variables
        q = casadi.SX.sym("q", nq)      # configuration (positions)
        v = casadi.SX.sym("v", nv)      # velocity
        u = casadi.SX.sym("u", nu)      # control
        dq_ = casadi.SX.sym("dq_", nv)  # velocity increment
        self.u_node = u
        self.q_node = q
        self.v_node = v
        self.dq_ = dq_

        B = np.array([1, 0])            # actuation matrix, first DoF is actuated since it is moving the cart
        tau = B @ u                     # robot’s generalized forces/torques
        a = cpin.aba(self.cmodel, self.cdata, q, v, tau) # Articulated Body Algorithm
        self.acc = a
        self.acc_func = casadi.Function("acc", [q, v, u], [a], ["q", "v", "u"], ["a"]) # create Casadi function for acceleration

    def create_discrete_dynamics(self):
        """
        Create the map `(q,v) -> (qnext, vnext)` using semi-implicit Euler integration.
        """
        q = self.q_node
        v = self.v_node
        u = self.u_node
        dq_ = self.dq_
        # q' = q + dq
        q_dq = cpin.integrate(self.cmodel, q, dq_)
        self.q_dq = q_dq
        # express acceleration using q' = q + dq
        a = self.acc_func(q_dq, v, u)

        dt = self.timestep
        vnext = v + a * dt
        qnext = cpin.integrate(self.cmodel, self.q_dq, dt * vnext)

        self.dyn_qv_fn_ = casadi.Function(
            "discrete_dyn",
            [q, dq_, v, u],
            [qnext, vnext],
            ["q", "dq_", "v", "u"],
            ["qnext", "vnext"],
        )

    def forward(self, x, u): # Current state and input  -> next state
        nq = self.model.nq
        nv = self.model.nv
        q = x[:nq] # Filter out position
        v = x[nq:] # Filter out velocity
        dq_ = np.zeros(nv)
        qnext, vnext = self.dyn_qv_fn_(q, dq_, v, u) # Feeding the dynamics function(Casadi function)
        xnext = np.concatenate((qnext, vnext)) # Concatenate position and velocity to form next state
        return xnext

    def residual_fwd(self, x, u, xnext): # Not used in this example
        nv = self.model.nv
        dq = np.zeros(nv)
        dqn = dq
        res = self.dyn_residual(x, u, xnext, dq, dqn)
        return res

# To-do
# The PinocchioCasadi class can be reused to convert from Pinocchio to casadi. (Also convert from semi-implicit Euler to explicit Euler)
# The visual and collision models can be used for debugging but for nothing else. 
# The make_cartpole function can be modified

class CartpoleDynamics(PinocchioCasadi):
    def __init__(self, timestep=0.05):
        model, collision_model, visual_model = make_cartpole()
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep)

#-------------------------------
# Actual code to run 

dt = 0.02 # time step, should match the MPC timestep
cartpole = CartpoleDynamics(timestep=dt)
model = cartpole.model

# ----------------------------
# I think the code ends here, the rest is just testing and visualization
# Just pass this model to ACADOS. 

print(model)

q0 = np.array([0.0, 0.95, 0.01])
q0 = pin.normalize(model, q0)
v = np.zeros(model.nv)
u = np.zeros(1)
a0 = cartpole.acc_func(q0, v, u)

print("a0:", a0)

x0 = np.append(q0, v)
xnext = cartpole.forward(x0, u)


def integrate_no_control(x0, nsteps):
    states_ = [x0.copy()]
    for t in range(nsteps):
        u = np.zeros(1)
        xnext = cartpole.forward(states_[t], u).ravel()
        states_.append(xnext)
    return states_


states_ = integrate_no_control(x0, nsteps=400)
states_ = np.stack(states_).T


# ------------------------
# Maybe add this bit of code for debugging purposes.

try:
    viz = MeshcatVisualizer(
        model=model,
        collision_model=cartpole.collision_model,
        visual_model=cartpole.visual_model,
    )

    viz.initViewer()
    viz.loadViewerModel("pinocchio")

    qs_ = states_[: model.nq, :].T
    viz.play(q_trajectory=qs_, dt=dt)
except ImportError as err:
    print(
        "Error while initializing the viewer. "
        "It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)