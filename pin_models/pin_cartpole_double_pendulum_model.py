import hppfcl as fcl
import numpy as np
import pinocchio as pin
from pin_models.pin_base_class import PinocchioCasadi

def make_cartpole_double_pendulum(config):
    model_config = config["model"]
    model = pin.Model()

    m1 = model_config["mass"]["pendulum1"]
    m2 = model_config["mass"]["pendulum2"]
    length = 0.8

    base_sizes = (0.4, 0.2, 0.05)

    # Create Joint for cart
    base = pin.JointModelPX() # PX: prismatic joint along x-axis
    base_id = model.addJoint(0, base, pin.SE3.Identity(), "base")

    # Inertia for cart
    base_inertia = pin.Inertia.FromBox(m1, *base_sizes) # mass, lx, ly, lz

    # Create Joint for first pendulum
    pendulum1_joint = pin.JointModelRY()
    pendulum1_id = model.addJoint(base_id, pendulum1_joint, pin.SE3.Identity(), "pendulum1")

    # Inertia for first pendulum
    pendulum1_inertia = pin.Inertia.FromSphere(m1, 0.1)

    # Create Joint for second pendulum
    pendulum2_joint = pin.JointModelRY()
    pendulum2_joint_pl = pin.SE3.Identity()
    pendulum2_joint_pl.translation = np.array([0.0, 0.0, length])
    pendulum2_id = model.addJoint(pendulum1_id, pendulum2_joint, pendulum2_joint_pl, "pendulum2")

    # Inertia for second pendulum
    pendulum2_inertia = pin.Inertia.FromSphere(m2, 0.1)

    # Place cart in 3D
    base_body_pl = pin.SE3.Identity() # base body placement (translation and rotation)

    model.appendBodyToJoint(base_id, base_inertia, base_body_pl) # attach body to joint

    # Place first pendulum in 3D
    pendulum1_body_pl = pin.SE3.Identity() # pendulum body placement (translation and rotation)
    pendulum1_body_pl.translation = np.array([0.0, 0.0, length]) # move pendulum up by length

    pole1_body_pl = pin.SE3.Identity() # pole body placement (translation and rotation)
    pole1_body_pl.translation = np.array([0.0, 0.0, length / 2]) # move pole up by length/2

    model.appendBodyToJoint(pendulum1_id, pendulum1_inertia, pendulum1_body_pl) # attach body to joint

    # Place second pendulum in 3D
    pendulum2_body_pl = pin.SE3.Identity() # pendulum body placement (translation and rotation)
    pendulum2_body_pl.translation = np.array([0.0, 0.0, length]) # move pendulum up by length

    pole2_body_pl = pin.SE3.Identity() # pole body placement (translation and rotation)
    pole2_body_pl.translation = np.array([0.0, 0.0, (length / 2)]) # move pole up by length/2

    model.appendBodyToJoint(pendulum2_id, pendulum2_inertia, pendulum2_body_pl) # attach body to joint

    # Make visual/collision models (not used in dynamics)
    collision_model = pin.GeometryModel()
    shape_base = fcl.Box(*base_sizes)
    radius = 0.01
    shape_pole = fcl.Capsule(radius, length)
    radius_pend = 0.1
    shape_pend = fcl.Sphere(radius_pend)
    RED_COLOR = np.array([1, 0.0, 0.0, 1.0])
    WHITE_COLOR = np.array([1, 1.0, 1.0, 1.0])

    geom_base = pin.GeometryObject("link_base", base_id, base_body_pl, shape_base)
    geom_base.meshColor = WHITE_COLOR

    geom_pole1 = pin.GeometryObject("link_pole1", pendulum1_id,  pole1_body_pl, shape_pole)
    geom_pole1.meshColor = WHITE_COLOR
    geom_pend1 = pin.GeometryObject("link_pend1", pendulum1_id, pendulum1_body_pl, shape_pend)
    geom_pend1.meshColor = RED_COLOR

    geom_pole2 = pin.GeometryObject("link_pole2", pendulum2_id,  pole2_body_pl, shape_pole)
    geom_pole2.meshColor = WHITE_COLOR
    geom_pend2 = pin.GeometryObject("link_pend2", pendulum2_id, pendulum2_body_pl, shape_pend)
    geom_pend2.meshColor = RED_COLOR

    collision_model.addGeometryObject(geom_base)
    collision_model.addGeometryObject(geom_pole1)
    collision_model.addGeometryObject(geom_pend1)
    collision_model.addGeometryObject(geom_pole2)
    collision_model.addGeometryObject(geom_pend2)
    visual_model = collision_model

    return model, collision_model, visual_model

class CartpoleDoublePendulumDynamics(PinocchioCasadi):
    def __init__(self, timestep: float, config):
        model, collision_model, visual_model = make_cartpole_double_pendulum(config)
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep, config=config)