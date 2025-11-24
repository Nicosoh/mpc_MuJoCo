import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from pin_exporter import export_ode_model
import scipy.linalg
from casadi import vertcat
import casadi as ca
from IK import generate_reference_trajectory

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
    dist = ca.sqrt(d2_seg)
    penetration = dist - (r1 + r2)

    # squared = convex and MPC-friendly
    d_capsule_sq = ca.fmax(penetration, 0)**2

    return ca.Function(
        "capsule_dist_sq",
        [p1, p2, q1, q2, r1, r2],
        # [d_capsule_sq],
        [penetration**2],
        ["p1", "p2", "q1", "q2", "r1", "r2"],
        ["dist_sq"]
    )
def build_capsule_collision_constraints(robot_sys, links, obstacles, collision_pairs):
    constraints = []

    for link_name, obs_name in collision_pairs:

        # -------------------------
        #     LINK CAPSULE
        # -------------------------
        link = links[link_name]

        # resolve symbolic endpoints from pin_model
        p1 = getattr(robot_sys, link["from"])   # CasADi 3-vector
        p2 = getattr(robot_sys, link["to"])     # CasADi 3-vector
        r1 = link["radius"]

        # -------------------------
        #     OBSTACLE CAPSULE
        # -------------------------
        obs = obstacles[obs_name]
        q1 = ca.SX(obs["from"])
        q2 = ca.SX(obs["to"])
        r2 = obs["radius"]

        # -------------------------
        #  SQUARED SEGMENT DISTANCE
        # -------------------------
        d2, Ps, Qt = segment_segment_squared_distance(p1, p2, q1, q2)

        # -------------------------
        #  HARD CONSTRAINT: d² ≥ (r1+r2)²
        # -------------------------
        min_dist_sq = (r1 + r2)**2
        constraints.append(d2 - min_dist_sq)

    return ca.vertcat(*constraints) if constraints else ca.SX([])

def setup(config, yref, collision_config=None):
    mpc_config = config["mpc"]

    Fmax = mpc_config["Fmax"]
    N_horizon = mpc_config["N_horizon"]
    RTI = mpc_config["use_RTI"]
    x0 = np.array(mpc_config["x0"])
    Tf = N_horizon * mpc_config["mpc_timestep"]  # Time horizon
    Q_mat = np.diag(mpc_config["Q_mat"]) # State cost weight matrix
    R_mat = np.diag(mpc_config["R_mat"]) # Input cost weight matrix

    # None for all other cases does does not require IK
    yref_traj_x = None
    
    # Create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Call model creation function
    model, robot_sys = export_ode_model(config)
    ocp.model = model

    # Generate reference trajectory and set collision constraints
    if mpc_config["IK_required"] and collision_config is not None:
        # Generate reference trajectory with IK

        # requires pin_model, not cmodel, yref, and obstacles, and config.
        yref_traj_q0 = generate_reference_trajectory(yref, collision_config["obstacles"], config)

        # Zero pad velocities to form full state      
        yref_traj_v0 = np.zeros_like(yref_traj_q0)
        yref_traj_x = np.hstack([yref_traj_q0, yref_traj_v0])
        config["mpc"]["x0"] = yref_traj_x[0] # Replace value in config so the simulator can use the value

        # Reassign x0 as first item
        x0 = yref_traj_x[0]

        # Generate collision constraints
        constraints = build_capsule_collision_constraints(robot_sys, 
                                                              collision_config["links"], 
                                                              collision_config["obstacles"], 
                                                              collision_config["collision_pairs"])
            
        ocp.model.con_h_expr = constraints
        ocp.constraints.lh = np.array([0.05])       # epsilon (additional safety distance)
        ocp.constraints.uh = np.array([1e10])

    # Extract state and input dimensions
    nx = model.x.rows() # Possible to extract the last predicted state here to insert into NN. maybe...
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'     # Stage cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'   # Terminal cost

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)  # Stage cost includes both states and input penalty
    ocp.cost.W_e = Q_mat                                # Terminal cost only inlcudes states

    if mpc_config["IK_required"] and collision_config is not None:

        ocp.model.cost_y_expr = vertcat(model.x, model.u)   # Stage cost includes both states and input
        ocp.model.cost_y_expr_e = model.x                   # Terminal cost only inlcudes states
        ocp.cost.yref  = np.zeros((ny, ))                    # Set stage references to match first entry of yref for all states and inputs
        ocp.cost.yref_e = np.zeros((ny_e, ))                 # Set terminal reference to match first entry of yref for states only
    else:
        ocp.model.cost_y_expr = vertcat(model.x, model.u)   # Stage cost includes both states and input
        ocp.model.cost_y_expr_e = model.x                   # Terminal cost only inlcudes states
        ocp.cost.yref  = yref[0, 1:ny+1]                    # Set stage references to match first entry of yref for all states and inputs
        ocp.cost.yref_e = yref[0, 1:ny_e+1]                 # Set terminal reference to match first entry of yref for states only

    # Set input constraints
    ocp.constraints.lbu = -np.array(Fmax)
    ocp.constraints.ubu = np.array(Fmax)
    # Apply above to all inputs
    ocp.constraints.idxbu = np.arange(nu)

    # Set initial constraint
    ocp.constraints.x0 = x0

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
    # ocp.solver_options.adaptive_levenberg_marquardt_lam

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        ocp.solver_options.nlp_solver_max_iter = 150
    
    ocp.solver_options.qp_solver_cond_N = N_horizon

    # Create solver based on settings above
    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json, verbose=False)

    return acados_ocp_solver, config, yref_traj_x

def get_reference_for_horizon(traj, t, N, mpc_dt, nu):
    """
    Build Acados-compatible reference arrays over the horizon.

    Args:
        traj: ndarray shaped (T, nx)
        t: current continuous time [s]
        N: horizon length
        mpc_dt: MPC step size
        nu: dimension of control input

    Returns:
        yref_stage: list of N vectors, each shape (nx + nu,)
        yref_terminal: vector shape (nx,)
    """

    T, nx = traj.shape

    # convert continuous time to discrete index
    start_idx = int(np.round(t / mpc_dt))

    # indices for 0..N, clipped inside bounds
    idxs = start_idx + np.arange(N + 1)
    idxs = np.clip(idxs, 0, T - 1)

    # extract states
    xrefs = traj[idxs]

    # build stage references (x,u) for k = 0..N-1
    yref_stage = []
    for k in range(N):
        xk = xrefs[k]
        uk = np.zeros(nu)              # zero input ref
        yref_stage.append(np.hstack((xk, uk)))

    # terminal reference (state-only)
    yref_terminal = xrefs[-1]

    return yref_stage, yref_terminal

class BaseMPCController:
    def __init__(self, config, yref, collision_config=None):
        # Setup MPC solver
        self.ocp_solver, self.config, self.yref_traj_x = setup(config, yref, collision_config)
        self.nx = self.ocp_solver.acados_ocp.dims.nx
        self.nu = self.ocp_solver.acados_ocp.dims.nu
        self.N = self.ocp_solver.acados_ocp.dims.N

        # Extract parameters from config
        self.use_RTI = config["mpc"]["use_RTI"]
        x0 = np.array(config["mpc"]["x0"])
        self.IK_required = config["mpc"]["IK_required"]
        self.mpc_timestep = config["mpc"]["mpc_timestep"]

        # Warm start
        for _ in range(5):
            self.ocp_solver.solve_for_x0(x0_bar=x0, fail_on_nonzero_status=False)

    def __call__(self, x, yref_now, full_traj, time):
        """Compute MPC input given MuJoCo state."""
        # Set yref
        if self.IK_required:
            yref_stage, yref_terminal = get_reference_for_horizon(self.yref_traj_x, time, self.N, self.mpc_timestep, self.nu)

            for stage in range(self.N):
                self.ocp_solver.cost_set(stage, "yref", yref_stage[stage], api='new')
            self.ocp_solver.cost_set(self.N, "yref", yref_terminal, api='new')  # Terminal reference (only x)
        else: 
            for stage in range(self.N):
                self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')
            self.ocp_solver.cost_set(self.N, "yref", yref_now[:self.nx], api='new')  # Terminal reference (only x)

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
            u = self.ocp_solver.solve_for_x0(x0_bar=x) # It is ok to fail during the warmup phase
        
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
        
        cost = self.ocp_solver.get_cost()

        return u, cost, qpos_traj, qvel_traj, u_traj