import scipy.linalg

import numpy as np
import casadi as ca

from casadi import vertcat
from acados_template import AcadosOcp, AcadosOcpSolver
from pin_exporter import export_ode_model

from neural_network.torch_exporter import export_torch_model

def clamp(x, lo=0.0, hi=1.0):
    return ca.fmin(ca.fmax(x, lo), hi)

def segment_segment_squared_distance(p1, p2, q1, q2):
    """
    p1, p2, q1, q2 : CasADi SX or MX 3×1 vectors
    returns squared distance between segments P and Q
    """

    u = p2 - p1   # segment P direction
    v = q2 - q1   # segment Q direction
    w0 = p1 - q1

    a = ca.dot(u, u)
    b = ca.dot(u, v)
    c = ca.dot(v, v)
    d = ca.dot(u, w0)
    e = ca.dot(v, w0)

    denom = a*c - b*b

    # Compute s*
    s = (b*e - c*d) / denom
    s = clamp(s)

    # Compute t*
    t = (a*e - b*d) / denom
    t = clamp(t)

    # Compute closest points
    Ps = p1 + u * s
    Qt = q1 + v * t

    return ca.dot(Ps - Qt, Ps - Qt), Ps, Qt

def capsule_squared_distance_function():
    """
    Returns a CasADi function:
       f(p1,p2,q1,q2,r1,r2) = squared distance between two capsules
    """

    p1 = ca.SX.sym("p1", 3)
    p2 = ca.SX.sym("p2", 3)
    q1 = ca.SX.sym("q1", 3)
    q2 = ca.SX.sym("q2", 3)

    r1 = ca.SX.sym("r1")   # radius of capsule 1
    r2 = ca.SX.sym("r2")   # radius of capsule 2

    d2_seg, Ps, Qt = segment_segment_squared_distance(p1, p2, q1, q2)

    # true capsule distance = max(0, distance - r1 - r2)
    dist = d2_seg - (r1 + r2)**2

    return ca.Function(
        "capsule_dist_sq",
        [p1, p2, q1, q2, r1, r2],
        [dist],
        ["p1", "p2", "q1", "q2", "r1", "r2"],
        ["dist_sq"]
    )

def build_capsule_collision_constraints(robot_sys, links, obstacles, collision_pairs):
    constraints = []

    for link_name, obs_name in collision_pairs:

        capsule_dist_sq = capsule_squared_distance_function()

        link = links[link_name]

        p1 = getattr(robot_sys, link["from"])   # CasADi 3-vector
        p2 = getattr(robot_sys, link["to"])     # CasADi 3-vector
        r1 = link["radius"]

        obs = obstacles[obs_name]
        q1 = obs["from"]
        q2 = obs["to"]
        r2 = obs["radius"]

        # q1 = np.array([1.1,0,0.5])
        # q2 = np.array([1.1,0,1.0])

        dist = capsule_dist_sq(p1, p2, q1, q2, r1, r2)

        constraints.append(dist)

        # # -------------------------
        # #     LINK CAPSULE
        # # -------------------------
        # link = links[link_name]

        # # resolve symbolic endpoints from pin_model
        # p1 = getattr(robot_sys, link["from"])   # CasADi 3-vector
        # p2 = getattr(robot_sys, link["to"])     # CasADi 3-vector
        # r1 = link["radius"]

        # # -------------------------
        # #     OBSTACLE CAPSULE
        # # -------------------------
        # obs = obstacles[obs_name]
        # # q1 = ca.SX(obs["from"])
        # # q2 = ca.SX(obs["to"])
        # # r2 = obs["radius"]
        # q1 = ca.SX(obs["from"])
        # q2 = ca.SX(obs["to"])
        # r2 = obs["radius"]
        
        # # -------------------------
        # #  SQUARED SEGMENT DISTANCE
        # # -------------------------
        # d2, Ps, Qt = segment_segment_squared_distance(p1, p2, q1, q2)

        # # -------------------------
        # #  HARD CONSTRAINT: d² ≥ (r1+r2)²
        # # -------------------------
        # min_dist_sq = (r1 + r2)**2
        # constraints.append(d2 - min_dist_sq)

    return vertcat(*constraints)

CONTROLLER_REGISTRY = {}

def register_controller(cls):
    CONTROLLER_REGISTRY[cls.__name__] = cls
    return cls

@register_controller
class BaseMPCController:
    def __init__(self, config, collision_config=None):
        # Extract parameters from config
        self.use_RTI = config["mpc"]["use_RTI"]
        self.x0 = np.array(config["mpc"]["x0"])
        self.mpc_timestep = config["mpc"]["mpc_timestep"]
        self.terminal_cost = config["mpc"]["terminal_cost"]
        self.IK_required = config["mpc"]["IK_required"]

        # Reassign x0 if IK is used
        if self.IK_required:
            self.x0 = np.array(config["mpc"]["x0_q"])

        # Setup MPC solver
        self.setup(config, collision_config)

        self.nx = self.ocp_solver.acados_ocp.dims.nx
        self.nu = self.ocp_solver.acados_ocp.dims.nu
        self.N = self.ocp_solver.acados_ocp.dims.N

        # Warm start
        for _ in range(5):
            self.ocp_solver.solve_for_x0(x0_bar=self.x0, fail_on_nonzero_status=True) # It is ok to fail during the warmup phase
       
    def setup(self, config, collision_config):
        mpc_config = config["mpc"]

        Fmax = mpc_config["Fmax"]
        N_horizon = mpc_config["N_horizon"]
        RTI = mpc_config["use_RTI"]
        x0 = np.array(mpc_config["x0"])
        Tf = N_horizon * mpc_config["mpc_timestep"]  # Time horizon
        Q_mat = np.diag(mpc_config["Q_mat"]) # State cost weight matrix
        R_mat = np.diag(mpc_config["R_mat"]) # Input cost weight matrix
        
        # Create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # Call model creation function
        model, robot_sys = export_ode_model(config)
        ocp.model = model

        # Extract state and input dimensions
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx

        # Set stage cost module
        ocp.cost.cost_type = 'NONLINEAR_LS'                 # Stage cost
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)  # Stage cost includes both states and input
        ocp.model.cost_y_expr = vertcat(model.x, model.u)   # Stage cost includes both states and input
        ocp.cost.yref  = np.zeros((ny, ))                   # Set stage references to match first entry of yref for all states and inputs

        # Set terminal cost module
        if self.terminal_cost:
            self.define_terminal_cost(ocp, model, config)
            
        # Generate collision constraints
        if collision_config is not None:
            self.add_hard_constraints(ocp, model, robot_sys, collision_config)

        # Set input constraints
        ocp.constraints.lbu = -np.array(Fmax)
        ocp.constraints.ubu = np.array(Fmax)

        # Apply above to all inputs
        ocp.constraints.idxbu = np.arange(nu)

        # Set initial constraint
        ocp.constraints.x0 = self.x0

        # set prediction horizon
        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = Tf # Total predicton time

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10
        ocp.solver_options.regularize_method = mpc_config["regularize_method"] # For the Hessian
        ocp.solver_options.levenberg_marquardt = mpc_config["levenberg_marquardt"]
        ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = mpc_config["nlp_solver_warm_start_first_qp_from_nlp"]
        ocp.solver_options.nlp_solver_warm_start_first_qp = mpc_config["nlp_solver_warm_start_first_qp"]
        ocp.solver_options.qp_solver_warm_start = mpc_config["qp_solver_warm_start"]

        if RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'
            ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
            ocp.solver_options.nlp_solver_max_iter = 150
        
        ocp.solver_options.qp_solver_cond_N = N_horizon

        # Create solver based on settings above
        solver_json = 'acados_ocp_' + model.name + '.json'
        self.ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json, verbose=False)
        
    def define_terminal_cost(self, ocp, model, config):
        Q_mat = np.diag(config["mpc"]["Q_mat"])
        ny_e = model.x.rows()
    
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = Q_mat                                # Terminal cost only inlcudes states
        ocp.model.cost_y_expr_e = model.x                   # Terminal cost only inlcudes states
        ocp.cost.yref_e = np.zeros((ny_e, ))                # Set terminal reference to match first entry of yref for states only
    
    def add_hard_constraints(self, ocp, model, robot_sys, collision_config):
        # Redefine in subclass
        pass

    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", yref_now[:self.nx], api='new')  # Terminal reference (only x)
    
    def collect_traj(self, full_traj):
        qpos_traj = []
        qvel_traj = []
        u_traj = []

        if full_traj: # Extract full state, control trajectories
            for i in range(self.N):
                xi = self.ocp_solver.get(i, "x")
                ui = self.ocp_solver.get(i, "u")
                qpos_traj.append(xi[:self.nx//2])
                qvel_traj.append(xi[self.nx//2:])
                u_traj.append(ui)

            # Get final state (at step N)
            xN = self.ocp_solver.get(self.N, "x")
            qpos_traj.append(xN[:self.nx//2])
            qvel_traj.append(xN[self.nx//2:])
        
        return qpos_traj, qvel_traj, u_traj

    def __call__(self, x, yref_now, full_traj):
        """Compute MPC input given MuJoCo state."""
        # Set yref
        self.set_yref(yref_now)

        if self.use_RTI:
            # Preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            
            status = self.ocp_solver.solve()
            if status != 0:
                raise RuntimeError("MPC solver returned status in RTI phase 1: ", status)

            # Set initial state
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            # Feedback phase
            self.ocp_solver.options_set('rti_phase', 2)

            status = self.ocp_solver.solve()
            if status != 0:
                raise RuntimeError("MPC solver returned status in RTI phase 2: ", status)

            # Get first control input
            u = self.ocp_solver.get(0, "u")

        else: # Without RTI
            # Solve ocp and get next control input
            u = self.ocp_solver.solve_for_x0(x0_bar=x)
        
        qpos_traj, qvel_traj, u_traj = self.collect_traj(full_traj)

        cost = self.ocp_solver.get_cost()

        return u, cost, qpos_traj, qvel_traj, u_traj

@register_controller
class NNMPCController(BaseMPCController):
    def __init__(self, config, collision_config=None):
        super().__init__(config, collision_config)

        if not config["mpc"]["terminal_cost"]:
            raise ValueError("NNMPCController requires terminal cost to be True.")
    
    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", np.zeros((1,)), api='new')  # Terminal reference (only x)
    
    def define_terminal_cost(self, ocp, model, config):
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = np.ones((1, 1))                      # Weights set to 1, meaning no scaling for the NN output
        ocp.cost.yref_e = np.zeros((1, ))                   # Set terminal reference to zero for NN output
        # Export trained NN model
        l4c_model = export_torch_model(config)
        # Evaluate NN symbolically
        y_sym = l4c_model(ca.transpose(model.x))
        ocp.model.cost_y_expr_e = y_sym
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = l4c_model.name

@register_controller
class ManipulatorMPCController(BaseMPCController):
    def __init__(self, config, collision_config=None):        
        super().__init__(config, collision_config)
        
        if not self.IK_required or collision_config is None:
            raise ValueError("ManipulatorMPCController requires IK and collision configuration.")
        
    def add_hard_constraints(self, ocp, model, robot_sys, collision_config):
        # Generate collision constraints
        # constraints = build_capsule_collision_constraints(robot_sys, 
        #                                                       collision_config["links"], 
        #                                                       collision_config["obstacles"], 
        #                                                       collision_config["collision_pairs"])

        # Add collision avoidance constraint between two capsules
        capsule_dist_sq = capsule_squared_distance_function()

        p1 = robot_sys.j_1
        p2 = robot_sys.attachment_site

        q1 = np.array([0.0, 0.7, 0.35])
        q2 = np.array([0.0, 0.7, 0.65])

        dist = capsule_dist_sq(p1, p2, q1, q2, 0.1, 0.05)

        ocp.model.con_h_expr = dist
        ocp.constraints.lh = np.array([0.05])       # epsilon (additional safety distance)
        ocp.constraints.uh = np.array([1e6])
    
    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now["stage"][stage], api='new')
        self.ocp_solver.cost_set(self.N, "yref", yref_now["terminal"][:self.nx], api='new')  # Terminal reference (only x)

@register_controller
class NNManipulatorMPCController(ManipulatorMPCController, NNMPCController):
    def __init__(self, config, collision_config=None):
        super().__init__(config, collision_config)