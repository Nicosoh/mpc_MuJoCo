import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper

class PinocchioCasadiRobotWrapper(RobotWrapper):
    """Take a Pinocchio model, turn it into a Casadi model
    and define the appropriate graphs.
    """
    def __init__(self, filename, config):
        super().initFromMJCF(filename=filename) # Parse MJCF file to RobotWrapper

        self.config = config
        pin_config = config["pin"]
        self.cmodel = cpin.Model(self.model)  # cast to CasADi model
        self.cdata = self.cmodel.createData() # create CasADi data
        self.timestep = config["mpc"]["mpc_timestep"]
        self.create_dynamics(pin_config)
        self.create_discrete_dynamics()
        if config["IK"]["IK_required"]:
            self.create_forward_kinematics()

    def create_dynamics(self, pin_config):
        """Create the acceleration expression and acceleration function."""
        nq = self.model.nq              # number of configuration variables
        nu = len(pin_config["actuated_joints"])
        nv = self.model.nv              # number of velocity variables
        q = casadi.SX.sym("q", nq)      # configuration (positions)
        v = casadi.SX.sym("v", nv)      # velocity
        u = casadi.SX.sym("u", nu)      # control
        self.u_node = u
        self.q_node = q
        self.v_node = v

        nu = len(pin_config["actuated_joints"])
        B = np.zeros((nq, nu))

        for i, joint_idx in enumerate(pin_config["actuated_joints"]):
            B[joint_idx, i] = 1.0
        
        tau = B @ u                     # robot’s generalized forces/torques
        a = cpin.aba(self.cmodel, self.cdata, q, v, tau) # Articulated Body Algorithm
        self.acc = a
        self.acc_func = casadi.Function("acc", [q, v, u], [a], ["q", "v", "u"], ["a"]) # create Casadi function for acceleration

    def create_discrete_dynamics(self):
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

    def create_forward_kinematics(self):
        """Create a function to compute forward kinematics."""
        # run forward kinematics symbolically
        cpin.forwardKinematics(self.cmodel, self.cdata, self.q_node)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        # CasADi FK expressions
        self.universe = self.cdata.oMf[self.cmodel.getFrameId("universe")].translation
        self.attachment_site = self.cdata.oMf[self.cmodel.getFrameId("attachment_site")].translation

        if self.config["model"]["name"] == "two_dof_arm":
            self.j_1 = self.cdata.oMf[self.cmodel.getFrameId("j_1")].translation
            self.j_2 = self.cdata.oMf[self.cmodel.getFrameId("j_2")].translation
        # elif self.config["model"]["name"] == "iiwa14"
        #     self.j_1 = self.cdata.oMf[self.cmodel.getFrameId("j_1")].translation
        #     self.j_2 = self.cdata.oMf[self.cmodel.getFrameId("j_2")].translation
        #                 self.j_1 = self.cdata.oMf[self.cmodel.getFrameId("j_1")].translation
        #     self.j_2 = self.cdata.oMf[self.cmodel.getFrameId("j_2")].translation
        #                 self.j_1 = self.cdata.oMf[self.cmodel.getFrameId("j_1")].translation
        #     self.j_2 = self.cdata.oMf[self.cmodel.getFrameId("j_2")].translation
        #                 self.j_1 = self.cdata.oMf[self.cmodel.getFrameId("j_1")].translation

    def forward(self, x, u): # Current state and input  -> next state
        nq = self.model.nq
        nv = self.model.nv
        q = x[:nq] # Filter out position
        v = x[nq:] # Filter out velocity
        qnext, vnext = self.dyn_qv_fn_(q, v, u) # Feeding the dynamics function(Casadi function)
        xnext = np.concatenate((qnext, vnext)) # Concatenate position and velocity to form next state
        return xnext