from acados_template import AcadosModel
from pin_models.pin_pendulum_model import CartpoleDynamics
from casadi import SX, vertcat

def export_ode_model(config) -> AcadosModel:
    """
    Converts a Pinocchio-based model into an AcadosModel for OCP solving.
    """
    mpc_config = config["mpc"]
    model_config=config["model"]

    # Create selected model
    if config["model"]["name"].lower() == "cartpole":
        print("Only 'cartpole' model is currently supported with Pinocchio.")
        pin_model = CartpoleDynamics(timestep=mpc_config["mpc_timestep"], model_config=model_config)

    # elif config["model"]["name"].lower() == "cartpole":
    #     pin_model = CartpoleDynamics(timestep=mpc_config["mpc_timestep"], model_config=model_config)

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

    # store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'

    return model