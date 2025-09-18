import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
# from archive.pendulum_model import export_pendulum_ode_model
from pin_exporter import export_ode_model
import scipy.linalg
from casadi import vertcat

def setup(config):
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
    model = export_ode_model(config)
    ocp.model = model

    # Extract state and input dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'     # Stage cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'   # Terminal cost

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)  # Stage cost includes both states and input penalty
    ocp.cost.W_e = Q_mat                                # Terminal cost only inlcudes states

    ocp.model.cost_y_expr = vertcat(model.x, model.u)   # Stage cost includes both states and input
    ocp.model.cost_y_expr_e = model.x                   # Terminal cost only inlcudes states
    ocp.cost.yref  = np.zeros((ny, ))                   # Set stage references as zero for all states and inputs
    ocp.cost.yref_e = np.zeros((ny_e, ))                # Set terminal reference as zeros for states only

    # Set input constraints
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([Fmax])
    # Apply above to the first idx in u which is F
    ocp.constraints.idxbu = np.array([0])

    # Set initial constraint
    ocp.constraints.x0 = x0

    # set prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = Tf # Total predicton time

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        ocp.solver_options.nlp_solver_max_iter = 150

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # Create solver based on settings above
    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # Create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator

class AcadosMPCController:
    def __init__(self, config):
        mpc_config = config["mpc"]

        # Extract parameters from config
        self.use_RTI = mpc_config["use_RTI"]
        x0 = np.array(mpc_config["x0"])

        # Setup MPC solver
        self.ocp_solver, _ = setup(config)
        self.nx = self.ocp_solver.acados_ocp.dims.nx
        self.nu = self.ocp_solver.acados_ocp.dims.nu

        # Warm start
        for _ in range(5):
            self.ocp_solver.solve_for_x0(x0_bar=x0)

    def __call__(self, state):
        """Compute MPC input given MuJoCo state."""
        qpos = state["qpos"]
        qvel = state["qvel"]
        x = np.concatenate([qpos, qvel])  # match Acados model

        # === Set time-varying reference ===
        # if state["time"] < 3.0:
        #     yref = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # size ny = nx + nu
        # else:
        #     yref = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        # N = self.ocp_solver.acados_ocp.dims.N

        # for stage in range(N):
        #     self.ocp_solver.cost_set(stage, "yref", yref)
        # self.ocp_solver.cost_set(N, "y_ref", yref[:self.nx])  # Terminal reference (only x)
        # ==================================

        if self.use_RTI:
            # Preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            
            status = self.ocp_solver.solve()
            if status != 0:
                print("MPC solver returned status: ", status)

            # Set initial state
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            # Feedback phase
            self.ocp_solver.options_set('rti_phase', 2)

            status = self.ocp_solver.solve()
            if status != 0:
                print("MPC solver returned status: ", status)

            # Get first control input
            u = self.ocp_solver.get(0, "u")

        else: # Without RTI
            # Solve ocp and get next control input
            u = self.ocp_solver.solve_for_x0(x0_bar=x)

        cost = self.ocp_solver.get_cost()
        return u, cost