from acados_template import AcadosModel
from pin_models import *
from casadi import SX, vertcat
from pin_models.pin_base_class import PinocchioCasadiRobotWrapper
import os

def export_ode_model(config) -> AcadosModel:
    """
    Converts a Pinocchio-based model into an AcadosModel for OCP solving.
    """
    base_dir = "models_xml"
    model_name = config["model"]["name"].lower()  # e.g., "two_dof_arm"
    filename = os.path.join(base_dir, f"{model_name}.xml")

    try:
        # Load the robot using PinocchioCasadi
        robot_sys = PinocchioCasadiRobotWrapper(filename=filename, config=config)

    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{filename}' does not exist. Check your models_xml folder.")

    model_name = "robot_sys_ode"

    # Use already-created Casadi symbolic variables
    q = robot_sys.q_node
    v = robot_sys.v_node
    u = robot_sys.u_node
    nq = robot_sys.model.nq
    nv = robot_sys.model.nv

    # State Vector: x = [q; v]
    x = vertcat(q, v)

    # xdot = [v; a(q,v,u)]
    a = robot_sys.acc  # acceleration expression
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

    return model, robot_sys