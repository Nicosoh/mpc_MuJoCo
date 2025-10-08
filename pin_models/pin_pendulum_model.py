import hppfcl as fcl
import numpy as np
import pinocchio as pin
from pin_models.pin_base_class import PinocchioCasadi

def make_pendulum(config):
    model_config = config["model"]
    model = pin.Model()

    m1 = model_config["mass"]["pendulum"]
    length = 0.8

    # Create Joints
    pendulum = pin.JointModelRY()
    pendulum_id = model.addJoint(0, pendulum, pin.SE3.Identity(), "pendulum")

    # Inertias
    pendulum_inertia = pin.Inertia.FromSphere(m1, 0.1)

    # Place bodies in 3D
    pendulum_body_pl = pin.SE3.Identity() # pendulum body placement (translation and rotation)
    pendulum_body_pl.translation = np.array([0.0, 0.0, length]) # move pendulum up by length

    pole_body_pl = pin.SE3.Identity() # pole body placement (translation and rotation)
    pole_body_pl.translation = np.array([0.0, 0.0, length / 2]) # move pole up by length/2

    model.appendBodyToJoint(pendulum_id, pendulum_inertia, pendulum_body_pl) # attach body to joint

    # Frame ID
    pendulum_frame_id = model.getJointId("pendulum")

    # Make visual/collision models (not used in dynamics)
    collision_model = pin.GeometryModel()
    radius = 0.01
    shape_pole = fcl.Capsule(radius, length)
    radius_pend = 0.1
    shape_pend = fcl.Sphere(radius_pend)
    RED_COLOR = np.array([1, 0.0, 0.0, 1.0])
    WHITE_COLOR = np.array([1, 1.0, 1.0, 1.0])
    geom_pole = pin.GeometryObject("link_pole", pendulum_id, pendulum_frame_id, pole_body_pl, shape_pole)
    geom_pole.meshColor = WHITE_COLOR
    geom_pend = pin.GeometryObject("link_pend", pendulum_id, pendulum_frame_id, pendulum_body_pl, shape_pend)
    geom_pend.meshColor = RED_COLOR

    collision_model.addGeometryObject(geom_pole)
    collision_model.addGeometryObject(geom_pend)
    visual_model = collision_model

    return model, collision_model, visual_model

class PendulumDynamics(PinocchioCasadi):
    def __init__(self, timestep: float, config):
        model, collision_model, visual_model = make_pendulum(config)
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep, config=config)