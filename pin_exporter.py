from acados_template import AcadosModel
from pin_models import *
from casadi import SX, vertcat

def export_ode_model(config) -> AcadosModel:
    """
    Converts a Pinocchio-based model into an AcadosModel for OCP solving.
    """
    mpc_config = config["mpc"]

    # Create selected model
    if config["model"]["name"].lower() == "cartpole":
        pin_model = CartpoleDynamics(timestep=mpc_config["mpc_timestep"], config=config)

    elif config["model"]["name"].lower() == "pendulum":
        pin_model = PendulumDynamics(timestep=mpc_config["mpc_timestep"], config=config)

    elif config["model"]["name"].lower() == "iiwa14":
        pin_model = iiwa14Dynamics(timestep=mpc_config["mpc_timestep"], config=config)

    elif config["model"]["name"].lower() == "double_pendulum":
        pin_model = DoublePendulumDynamics(timestep=mpc_config["mpc_timestep"], config=config)
    
    elif config["model"]["name"].lower() == "cartpole_double_pendulum":
        pin_model = CartpoleDoublePendulumDynamics(timestep=mpc_config["mpc_timestep"], config=config)
    
    elif config["model"]["name"].lower() == "two_dof_arm":
        pin_model = TwoDOFArmDynamics(timestep=mpc_config["mpc_timestep"], config=config)

    else:
        raise ValueError(f"Unknown model name '{config['model']['name']}'. Add in elif statement for new models")

    model_name = "pin_model_ode"

    # Use already-created Casadi symbolic variables
    q = pin_model.q_node
    v = pin_model.v_node
    u = pin_model.u_node
    nq = pin_model.model.nq
    nv = pin_model.model.nv

    # State Vector: x = [q; v]
    x = vertcat(q, v)

    # xdot = [v; a(q,v,u)]
    a = pin_model.acc  # acceleration expression
    xdot = vertcat(v, a)

    # Explicit dynamics
    f_expl = xdot

    # Implicit dynamics
    xdot_sym = SX.sym("xdot", nq + nv)
    f_impl = xdot_sym - f_expl

    # Wrap into AcadosModel
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot_sym
    model.u = u
    model.name = model_name

    return model, pin_model