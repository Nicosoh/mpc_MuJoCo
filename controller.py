import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from pin_exporter import export_ode_model
import scipy.linalg
from casadi import vertcat

def setup(config, yref):
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
    nx = model.x.rows() # Possible to extract the last predicted state here to insert into NN. maybe...
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
    ocp.cost.yref  = yref[0, 1:ny+1]                        # Set stage references to match first entry of yref for all states and inputs
    ocp.cost.yref_e = yref[0, 1:ny_e+1]                 # Set terminal reference to match first entry of yref for states only
    # ocp.cost.yref  = np.zeros((ny, ))                   # Set stage references as zero for all states and inputs
    # ocp.cost.yref_e = np.zeros((ny_e, ))                # Set terminal reference as zeros for states only

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

class BaseMPCController:
    def __init__(self, config, yref):
        mpc_config = config["mpc"]

        # Extract parameters from config
        self.use_RTI = mpc_config["use_RTI"]
        x0 = np.array(mpc_config["x0"])

        # Setup MPC solver
        self.ocp_solver, _ = setup(config, yref)
        self.nx = self.ocp_solver.acados_ocp.dims.nx
        self.nu = self.ocp_solver.acados_ocp.dims.nu
        self.N = self.ocp_solver.acados_ocp.dims.N

        # Warm start
        for _ in range(5):
            self.ocp_solver.solve_for_x0(x0_bar=x0)
    
    # def get_full_OCP(self, state):
    #     """Compute MPC input and return full trajectory and control sequence."""
    #     qpos = state["qpos"]
    #     qvel = state["qvel"]
    #     x = np.concatenate([qpos, qvel])  # match Acados model

    #     if self.use_RTI:
    #         # RTI preparation phase
    #         self.ocp_solver.options_set('rti_phase', 1)
    #         status = self.ocp_solver.solve()
    #         if status != 0:
    #             print("MPC solver returned status in RTI phase 1: ", status)

    #         self.ocp_solver.set(0, "lbx", x)
    #         self.ocp_solver.set(0, "ubx", x)

    #         # RTI feedback phase
    #         self.ocp_solver.options_set('rti_phase', 2)
    #         status = self.ocp_solver.solve()
    #         if status != 0:
    #             print("MPC solver returned status in RTI phase 2: ", status)
    #     else:
    #         # Without RTI
    #         self.ocp_solver.solve_for_x0(x0_bar=x)

    #     # Extract full state and control trajectories
    #     x_traj = []
    #     u_traj = []

    #     for i in range(self.N):
    #         xi = self.ocp_solver.get(i, "x")
    #         ui = self.ocp_solver.get(i, "u")
    #         x_traj.append(xi)
    #         u_traj.append(ui)

    #     # Get final state (at step N)
    #     xN = self.ocp_solver.get(self.N, "x")
    #     x_traj.append(xN)
        
    #     # Full trajs, control input
    #     return np.array(x_traj), np.array(u_traj)

    def __call__(self, x, yref_now, full_traj):
        """Compute MPC input given MuJoCo state."""
        # Set yref
        for stage in range(self.N):
            self.ocp_solver.cost_set(stage, "yref", yref_now, api='new')
        self.ocp_solver.cost_set(self.N, "y_ref", yref_now[:self.nx], api='new')  # Terminal reference (only x)

        if self.use_RTI:
            # Preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            
            status = self.ocp_solver.solve()
            if status != 0:
                print("MPC solver returned status in RTI phase 1: ", status)

            # Set initial state
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            # Feedback phase
            self.ocp_solver.options_set('rti_phase', 2)

            status = self.ocp_solver.solve()
            if status != 0:
                print("MPC solver returned status in RTI phase 2: ", status)

            # Get first control input
            u = self.ocp_solver.get(0, "u")

        else: # Without RTI
            # Solve ocp and get next control input
            u = self.ocp_solver.solve_for_x0(x0_bar=x)

        x_traj = []
        u_traj = []

        if full_traj: # Extract full state, control trajectories
            for i in range(self.N):
                xi = self.ocp_solver.get(i, "x")
                ui = self.ocp_solver.get(i, "u")
                x_traj.append(xi)
                u_traj.append(ui)

            # Get final state (at step N)
            xN = self.ocp_solver.get(self.N, "x")
            x_traj.append(xN)
        
        cost = self.ocp_solver.get_cost()

        return u, cost, x_traj, u_traj