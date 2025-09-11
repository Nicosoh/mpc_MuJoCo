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

    def __init__(self, model: pin.Model, timestep: float):
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
            [q, v, u],
            [qnext, vnext],
            ["q", "v", "u"],
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
    
class CartpoleDynamics(PinocchioCasadi):
    def __init__(self, timestep: float):
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

# -----------------------------
# Example from chatgpt
import pinocchio as pin
from pinocchio.utils import zero
import numpy as np
from casadi import SX, vertcat, Function

from acados_template import AcadosModel

def export_double_cartpole_model_pinocchio():
    """
    Double cart-pole model using Pinocchio numerical dynamics, 
    wrapped as CasADi function for ACADOS.
    """

    # ----------------------
    # 1. Build Pinocchio model
    model = pin.Model()

    # Cart joint (prismatic along x)
    joint_cart = model.addJoint(0, pin.JointModelPX(), pin.SE3.Identity(), "cart")

    # First pendulum (revolute around z)
    l1, m1 = 0.8, 0.1
    joint1 = model.addJoint(joint_cart, pin.JointModelRZ(), pin.SE3.Identity(), "pend1")
    inertia1 = pin.Inertia(m1, np.zeros(3), np.diag([0,0,(l1*l1)/12]))
    model.appendBodyToJoint(joint1, inertia1, pin.SE3.Identity())

    # Second pendulum (revolute around z)
    l2, m2 = 0.8, 0.1
    joint2 = model.addJoint(joint1, pin.JointModelRZ(), pin.SE3(np.eye(3), np.array([l1,0,0])), "pend2")
    inertia2 = pin.Inertia(m2, np.zeros(3), np.diag([0,0,(l2*l2)/12]))
    model.appendBodyToJoint(joint2, inertia2, pin.SE3.Identity())

    data = model.createData()

    # ----------------------
    # 2. Define ACADOS states and input
    nq = model.nq
    nv = model.nv

    q = SX.sym('q', nq)
    v = SX.sym('v', nv)
    u = SX.sym('u', 1)  # F input

    # ----------------------
    # 3. Define a function that wraps Pinocchio forward dynamics
    # Only numerical: will need to use MX or callback in ACADOS
    def pinocchio_forward(q_val, v_val, u_val):
        qn = np.array(q_val).flatten()
        vn = np.array(v_val).flatten()
        tau = np.zeros(model.nv)
        tau[0] = u_val[0]  # only force on cart
        pin.forwardDynamics(model, data, qn, vn, tau)
        return data.qdd

    # ----------------------
    # 4. CasADi wrapper using SX.sym placeholder
    # Note: in ACADOS we usually implement this as an external function
    qdd = SX.sym('qdd', nv)  # placeholder
    xdot = vertcat(v, qdd)

    # ----------------------
    # 5. Build ACADOS model
    model_acados = AcadosModel()
    model_acados.x = vertcat(q, v)
    model_acados.xdot = xdot
    model_acados.u = u
    model_acados.f_expl_expr = xdot
    model_acados.name = 'double_cartpole_pinocchio'

    return model_acados
