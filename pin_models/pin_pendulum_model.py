import casadi
import hppfcl as fcl
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

def make_cartpole(model_config):
    model = pin.Model()

    m1 = model_config["mass"]["cart"]
    m2 = model_config["mass"]["pendulum"]
    length = 0.8

    base_sizes = (0.4, 0.2, 0.05)

    # Create Joints
    base = pin.JointModelPX() # PX: prismatic joint along x-axis
    base_id = model.addJoint(0, base, pin.SE3.Identity(), "base")

    pendulum = pin.JointModelRY()
    pendulum_id = model.addJoint(1, pendulum, pin.SE3.Identity(), "pendulum")

    # Inertias
    base_inertia = pin.Inertia.FromBox(m1, *base_sizes) # mass, lx, ly, lz
    pendulum_inertia = pin.Inertia.FromSphere(m2, 0.1)

    # Place bodies in 3D
    base_body_pl = pin.SE3.Identity() # base body placement (translation and rotation)
    
    pendulum_body_pl = pin.SE3.Identity() # pendulum body placement (translation and rotation)
    pendulum_body_pl.translation = np.array([0.0, 0.0, length]) # move pendulum up by length

    pole_body_pl = pin.SE3.Identity() # pole body placement (translation and rotation)
    pole_body_pl.translation = np.array([0.0, 0.0, length / 2]) # move pole up by length/2

    model.appendBodyToJoint(base_id, base_inertia, base_body_pl) # attach body to joint
    model.appendBodyToJoint(pendulum_id, pendulum_inertia, pendulum_body_pl) # attach body to joint

    # make visual/collision models (not used in dynamics)
    collision_model = pin.GeometryModel()
    shape_base = fcl.Box(*base_sizes)
    radius = 0.01
    shape_pole = fcl.Capsule(radius, length)
    radius_pend = 0.1
    shape_pend = fcl.Sphere(radius_pend)
    RED_COLOR = np.array([1, 0.0, 0.0, 1.0])
    WHITE_COLOR = np.array([1, 1.0, 1.0, 1.0])
    geom_base = pin.GeometryObject("link_base", base_id, shape_base, base_body_pl)
    geom_base.meshColor = WHITE_COLOR
    geom_pole = pin.GeometryObject("link_pole", pendulum_id, shape_pole, pole_body_pl)
    geom_pole.meshColor = RED_COLOR
    geom_pend = pin.GeometryObject("link_pend", pendulum_id, shape_pend, pendulum_body_pl)
    geom_pend.meshColor = RED_COLOR

    collision_model.addGeometryObject(geom_base)
    collision_model.addGeometryObject(geom_pole)
    collision_model.addGeometryObject(geom_pend)
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
        self.create_discrete_dynamics2()

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

    def create_discrete_dynamics2(self):
        """
        Create the map `(q,v) -> (qnext, vnext)` using explicit Euler integration.
        """
        q = self.q_node
        v = self.v_node
        u = self.u_node

        dt = self.timestep

        # Compute acceleration
        a = self.acc_func(q, v, u)

        # Explicit Euler integration
        qnext = cpin.integrate(self.cmodel, q, dt * v)  # use current velocity
        vnext = v + a * dt                              # use current acceleration

        # Define CasADi function for discrete dynamics
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
    def __init__(self, timestep: float, model_config):
        model, collision_model, visual_model = make_cartpole(model_config)
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep)