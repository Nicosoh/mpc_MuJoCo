from pin_models.pin_base_class import PinocchioCasadiRobotWrapper
import hppfcl as fcl
import numpy as np
import pinocchio as pin

def make_cartpole(config):
    model = pin.Model()
    
    model_config = config["model"]
    m1 = model_config["mass"]["cart"]
    m2 = model_config["mass"]["pendulum"]
    length = 0.8

    base_sizes = (0.4, 0.2, 0.05)

    # Create Joints
    base = pin.JointModelPX() # PX: prismatic joint along x-axis
    base_id = model.addJoint(0, base, pin.SE3.Identity(), "base")

    pendulum = pin.JointModelRY()
    pendulum_id = model.addJoint(1, pendulum, pin.SE3.Identity(), "pendulum")

    # Inertias
    base_inertia = pin.Inertia.FromBox(m1, *base_sizes) # mass, lx, ly, lz
    pendulum_inertia = pin.Inertia.FromSphere(m2, 0.1)

    # Place bodies in 3D
    base_body_pl = pin.SE3.Identity() # base body placement (translation and rotation)
    
    pendulum_body_pl = pin.SE3.Identity() # pendulum body placement (translation and rotation)
    pendulum_body_pl.translation = np.array([0.0, 0.0, length]) # move pendulum up by length

    pole_body_pl = pin.SE3.Identity() # pole body placement (translation and rotation)
    pole_body_pl.translation = np.array([0.0, 0.0, length / 2]) # move pole up by length/2

    model.appendBodyToJoint(base_id, base_inertia, base_body_pl) # attach body to joint
    model.appendBodyToJoint(pendulum_id, pendulum_inertia, pendulum_body_pl) # attach body to joint

    # make visual/collision models (not used in dynamics)
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
    geom_pole = pin.GeometryObject("link_pole", pendulum_id, pole_body_pl, shape_pole)
    geom_pole.meshColor = RED_COLOR
    geom_pend = pin.GeometryObject("link_pend", pendulum_id, pendulum_body_pl, shape_pend)
    geom_pend.meshColor = RED_COLOR

    collision_model.addGeometryObject(geom_base)
    collision_model.addGeometryObject(geom_pole)
    collision_model.addGeometryObject(geom_pend)
    visual_model = collision_model

    return model, collision_model, visual_model
    
class CartpoleDynamics(PinocchioCasadiRobotWrapper):
    def __init__(self, timestep: float, config):
        model, collision_model, visual_model = make_cartpole(config)
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep, config=config)