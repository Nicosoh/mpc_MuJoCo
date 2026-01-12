import scipy.linalg

import numpy as np
import casadi as ca

from utils import *
from casadi import vertcat, SX
from pin_exporter import export_ode_model
from acados_template import AcadosOcp, AcadosOcpSolver
from neural_network.torch_exporter import export_torch_model

CONTROLLER_REGISTRY = {}

def register_controller(cls):
    CONTROLLER_REGISTRY[cls.__name__] = cls
    return cls

@register_controller
class BaseMPCController:
    def __init__(self, config, collision_config=None):
        # Extract parameters from config
        self.config = config
        self.use_RTI = config["mpc"]["use_RTI"]
        self.x0 = np.array(config["mpc"]["x0"])
        self.mpc_timestep = config["mpc"]["mpc_timestep"]
        self.terminal_cost = config["mpc"]["terminal_cost"]
        self.IK_required = config["IK"]["IK_required"]

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
            self.ocp_solver.solve_for_x0(x0_bar=self.x0, fail_on_nonzero_status=False, print_stats_on_failure=False) # It is ok to fail during the warmup phase
       
    def setup(self, config, collision_config):
        mpc_config = config["mpc"]

        Fmax = mpc_config["Fmax"]
        N_horizon = mpc_config["N_horizon"]
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

        # Set stage cost module
        ocp.cost.cost_type = 'NONLINEAR_LS'                 # Stage cost
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)  # Stage cost includes both states and input
        ocp.model.cost_y_expr = vertcat(model.x, model.u)   # Stage cost includes both states and input
        ocp.cost.yref  = np.zeros((ny, ))                   # Set stage references to match first entry of yref for all states and inputs

        # Set terminal cost module
        if self.terminal_cost:
            self.define_terminal_cost(ocp, model, config)
            
        # Generate collision constraints
        if collision_config is not None and config["collision"]["collision_avoidance"]:
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

        if self.use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'
            ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
            ocp.solver_options.nlp_solver_max_iter = 100
        
        ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.solver_options.nlp_solver_tol_stat = 1e-3
        # Create solver based on settings above
        solver_json = 'acados_ocp_' + model.name + '.json'
        self.ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json, verbose=False)
        
    def define_terminal_cost(self, ocp, model, config):
        Q_mat = np.diag(self.config["mpc"]["Q_mat"])
        ny_e = model.x.rows()
    
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = Q_mat                                # Terminal cost only inlcudes states
        ocp.model.cost_y_expr_e = model.x                   # Terminal cost only inlcudes states
        ocp.cost.yref_e = np.zeros((ny_e, ))                # Set terminal reference to match first entry of yref for states only
    
    def add_hard_constraints(self, ocp, model, robot_sys, collision_config):
        raise NotImplementedError("add_hard_constraints method not implemented in BaseMPCController.")

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

        qpos_traj = np.array(qpos_traj)
        qvel_traj = np.array(qvel_traj)
        u_traj = np.array(u_traj)

        return qpos_traj, qvel_traj, u_traj
    
    def collect_cost(self, qpos_traj, qvel_traj, u_traj, yref_now):
        if qpos_traj.size == 0 or qvel_traj.size == 0 or u_traj.size == 0:
            raise ValueError("Empty trajectory detected, set full_traj=True.")

        # Full state matrix: X = [qpos | qvel]
        X = np.column_stack((qpos_traj, qvel_traj))  # shape (N+1, nx)
        U = u_traj                                   # shape (N, nu)

        stage_ref, terminal_ref = self.build_references(X, U, yref_now)

        stage_cost = self.compute_stage_cost(X, U, stage_ref)
        terminal_cost = self.compute_terminal_cost(X, terminal_ref)

        total_cost = stage_cost + terminal_cost

        self.check_solver_cost(total_cost)

        return stage_cost, terminal_cost, total_cost
    
    def build_references(self, X, U, yref_now):
        nx, nu = X.shape[1], U.shape[1]

        if isinstance(yref_now, dict):
            stage_ref = yref_now["stage"]
            terminal_ref = yref_now["terminal"][:nx]
        else:
            stage_ref = np.tile(yref_now, (self.N, 1))
            terminal_ref = yref_now[:nx]

        return stage_ref, terminal_ref
    
    def compute_stage_cost(self, X, U, stage_ref):
        Q = np.diag(self.config["mpc"]["Q_mat"])
        R = np.diag(self.config["mpc"]["R_mat"])

        nx, nu = X.shape[1], U.shape[1]

        XU = np.hstack([X[:self.N], U[:self.N]])
        e = XU - stage_ref

        Q_full = np.block([
            [Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), R]
        ])

        stage_cost = 0.5 * self.mpc_timestep * np.sum(
            np.sum((e @ Q_full) * e, axis=1)
        )
        return stage_cost
    
    def compute_terminal_cost(self, X, terminal_ref):
        if self.terminal_cost:
            Q = np.diag(self.config["mpc"]["Q_mat"])
            e = X[self.N] - terminal_ref
            return 0.5 * e @ Q @ e
        else:
            return 0
    
    def check_solver_cost(self, total_cost):
        solver_cost = self.ocp_solver.get_cost()
        if not np.isclose(solver_cost, total_cost, rtol=1e-6, atol=1e-8):
            raise RuntimeError(
                f"Cost mismatch:\nsolver={solver_cost}\ncomputed={total_cost}"
            )

    def __call__(self, x, yref_now, full_traj):
        """Compute MPC input given MuJoCo state."""
        # Set yref
        self.set_yref(yref_now)

        if self.use_RTI:
            # Preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            
            status = self.ocp_solver.solve()
            if status != 0:
                # raise RuntimeError("MPC solver returned status in RTI phase 1: ", status)
                raise RuntimeError(f"MPC solver returned status {status} in RTI phase 1")
            
            # Set initial state
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            # Feedback phase
            self.ocp_solver.options_set('rti_phase', 2)

            status = self.ocp_solver.solve()
            if status != 0:
                raise RuntimeError(f"MPC solver returned status {status} in RTI phase 2")

            # Get first control input
            u = self.ocp_solver.get(0, "u")

        # else: # Without RTI
            # Solve ocp and get next control input
            # u = self.ocp_solver.solve_for_x0(x0_bar=x, fail_on_nonzero_status=False, print_stats_on_failure=False)
            u = self.ocp_solver.solve_for_x0(x0_bar=x, fail_on_nonzero_status=False)
        
        else: # Without RTI (modofied solve_for_x0 to avoid min_step issue)
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            status = self.ocp_solver.solve()

            if status != 0 and status != 2:
                raise RuntimeError(f"MPC solver returned status {status} in SQP mode")

            u = self.ocp_solver.get(0, "u")
        
        # Collect traj
        qpos_traj, qvel_traj, u_traj = self.collect_traj(full_traj)

        # Collect cost components
        stage_cost, terminal_cost, total_cost = self.collect_cost(qpos_traj, qvel_traj, u_traj, yref_now)

        return u, stage_cost, terminal_cost, total_cost, qpos_traj, qvel_traj, u_traj

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
        self.l4c_model = export_torch_model(config)
        # Evaluate NN symbolically
        y_sym = self.l4c_model(ca.transpose(model.x))
        ocp.model.cost_y_expr_e = y_sym
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name
    
    def compute_terminal_cost(self, X, terminal_ref):
        # Terminal state (nx,)
        xN = X[self.N]

        # Evaluate NN numerically
        yN = np.asarray(self.l4c_model(xN)).squeeze()

        # NONLINEAR_LS terminal cost
        terminal_cost = 0.5 * yN**2
        return terminal_cost

@register_controller
class ManipulatorMPCController(BaseMPCController):
    def __init__(self, config, collision_config=None):        
        super().__init__(config, collision_config)
        
    def add_hard_constraints(self, ocp, model, robot_sys, collision_config):
        # Generate collision constraints
        constraints = build_capsule_collision_constraints(robot_sys, 
                                                              collision_config["links"], 
                                                              collision_config["obstacles"], 
                                                              collision_config["collision_pairs"])

        ocp.model.con_h_expr = constraints
        ocp.constraints.lh = np.zeros(constraints.shape[0])
        ocp.constraints.uh = 1e10 * np.ones(constraints.shape[0])

        ocp.model.con_h_expr_e = constraints
        ocp.constraints.lh_e = np.zeros(constraints.shape[0])
        ocp.constraints.uh_e = 1e10 * np.ones(constraints.shape[0])

    def set_yref(self, yref_now):
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now["stage"][stage], api='new')
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", yref_now["terminal"][:self.nx], api='new')  # Terminal reference (only x)

@register_controller
class NNManipulatorMPCController(ManipulatorMPCController, NNMPCController):
    def __init__(self, config, collision_config=None):
        super().__init__(config, collision_config)
    
    def define_terminal_cost(self, ocp, model, config):
        ocp.cost.cost_type_e = 'NONLINEAR_LS'               # Terminal cost
        ocp.cost.W_e = np.ones((1, 1))                      # Weights set to 1, meaning no scaling for the NN output
        ocp.cost.yref_e = np.zeros((1, ))                   # Set terminal reference to zero for NN output

        # Create parameters for goal, obstacle
        self.p = np.array(self.config["mpc"]["yref"])
        ocp.model.p = SX.sym('p', self.p.shape[0])  # Full yref as parameter
        ocp.parameter_values = np.zeros(self.p.shape[0])

        # Export trained NN model
        self.l4c_model = export_torch_model(config)
        # Evaluate NN symbolically
        y_sym = self.l4c_model(ca.transpose(vertcat(model.x, ocp.model.p)))
        ocp.model.cost_y_expr_e = y_sym
        # Link shared library
        ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name
    
    def set_yref(self, yref_now):
        # Stage
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now["stage"][stage], api='new')
            self.ocp_solver.set(stage, "p", self.p)                                             # Modify Goal/obstacle position

        # Terminal
        if self.terminal_cost:
            self.ocp_solver.cost_set(self.N, "yref", np.zeros((1,)), api='new')  # Terminal reference (only x)

        self.ocp_solver.set(self.N, "p", self.p)                                             # Modify Goal/obstacle position

    def compute_terminal_cost(self, X, terminal_ref):
        # Terminal state (nx,)
        xN = X[self.N]

        xN_p = np.concatenate([xN, self.p])

        # Evaluate NN numerically
        yN = np.asarray(self.l4c_model(xN_p)).squeeze()

        # NONLINEAR_LS terminal cost
        terminal_cost = 0.5 * yN**2
        return terminal_cost