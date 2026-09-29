#!/usr/bin/env python3
"""Franka repeatedly folds and unfolds two triangles about a passive hinge.

Run from the repository root:
    uv run python thesis/folding_demo.py --viz viser
    # Open http://localhost:8080 in your browser.

The gripper holds the moving panel with an ideal spherical pinch constraint.
Press Ctrl+C in the terminal to stop (closing a Viser tab leaves the server running).
"""

import argparse
import math

from fold_geometry import (
    Board,
    add,
    dot,
    hinge_angle,
    inverse_rotate,
    norm,
    rotate,
    scale,
    slerp,
    smoothstep,
    sub,
)


def move_panel(board, end_angle, seconds, dt, measured_angle, step, hold, down_q):
    """Move the held panel to an endpoint, waiting when physical motion lags."""
    start_angle = min(math.pi, max(0.0, measured_angle()))
    progress = 0.0
    elapsed = 0.0
    error = 0.0
    target_angle = start_angle
    while progress < 1.0:
        lag = abs(target_angle - measured_angle())
        if error < 0.025 and lag < math.radians(12):
            progress = min(1.0, progress + dt / seconds)
        target_angle = start_angle + (end_angle - start_angle) * smoothstep(progress)
        error = step(board.handle_world(target_angle), down_q, board.handle_radius)
        elapsed += dt
        if elapsed > 3 * seconds:
            raise RuntimeError(f"Motion stalled at {math.degrees(measured_angle()):.1f} degrees.")
    hold(board.handle_world(end_angle), down_q, 1.0, board.handle_radius)
    if abs(measured_angle() - end_angle) > math.radians(6):
        raise RuntimeError("The panel did not reach the commanded endpoint.")


def run(args, app):
    # Isaac/Omniverse imports must follow AppLauncher construction.
    import torch

    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
    from isaaclab.cloner import ReplicateSession
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

    board = Board(side=args.board_size)
    dt = 1.0 / 240.0
    # SimulationCfg defaults to PhysX on the checked develop revision.
    # use_fabric=False is intentional: this single-scene demo toggles a USD joint.
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args.device, use_fabric=False)
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    sim.set_camera_view((1.35, 1.25, 1.65), (0.40, 0.0, 0.75))

    def usd_quat(q):
        return Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3]))

    def transform(prim, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)):
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddOrientOp().Set(usd_quat(orientation))
        return xf

    def visual_material(name, color):
        mat = UsdShade.Material.Define(stage, "/World/Materials/" + name)
        shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    mats = {
        "table": visual_material("Table", (0.33, 0.27, 0.22)),
        "fixed": visual_material("FixedTriangle", (0.08, 0.36, 0.64)),
        "moving": visual_material("MovingTriangle", (0.94, 0.36, 0.08)),
        "metal": visual_material("HingeAndHandle", (0.68, 0.70, 0.73)),
        "ground": visual_material("Ground", (0.18, 0.20, 0.23)),
    }
    physics_mat = UsdShade.Material.Define(stage, "/World/Materials/Contact")
    material_api = UsdPhysics.MaterialAPI.Apply(physics_mat.GetPrim())
    material_api.CreateStaticFrictionAttr(0.6)
    material_api.CreateDynamicFrictionAttr(0.5)
    material_api.CreateRestitutionAttr(0.0)

    def finish_shape(shape, mat, collision=True):
        prim = shape.GetPrim()
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(mat)
        if collision:
            # Viser hides collider meshes when a body also has visual geometry.
            # Author a visible sibling and keep this shape exclusively for physics.
            visual_path = prim.GetPath().GetParentPath().AppendChild(prim.GetName() + "Visual")
            layer = stage.GetEditTarget().GetLayer()
            Sdf.CopySpec(layer, prim.GetPath(), layer, visual_path)
            UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(prim)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr(0.001)
            physx_collision.CreateRestOffsetAttr(0.0)
            binding.Bind(physics_mat, materialPurpose="physics")
        return prim

    def box(path, position, size, mat):
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.0)
        transform(shape.GetPrim(), position).AddScaleOp().Set(Gf.Vec3f(*size))
        finish_shape(shape, mat)
        return shape

    # These boxes have CollisionAPI but no RigidBodyAPI, so they are static.
    box("/World/Ground", (0.4, 0.0, -0.02), (2.6, 2.2, 0.04), mats["ground"])
    box("/World/Table", (0.40, 0.0, board.table_top / 2), (1.25, 1.0, board.table_top), mats["table"])
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(2200.0)

    board_root = "/World/Board"
    fixed_path = board_root + "/FixedTriangle"
    moving_path = board_root + "/MovingTriangle"
    UsdGeom.Xform.Define(stage, board_root)

    def make_panel(path, moving):
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        transform(root, board.origin)
        if moving:
            rigid = UsdPhysics.RigidBodyAPI.Apply(root)
            rigid.CreateRigidBodyEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.16)
            body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
            body_api.CreateSolverPositionIterationCountAttr(16)
            body_api.CreateSolverVelocityIterationCountAttr(4)
            body_api.CreateAngularDampingAttr(0.25)
        vertices, faces = board.prism(moving)
        mesh = UsdGeom.Mesh.Define(stage, path + "/Panel")
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(*(min(v[k] for v in vertices) for k in range(3))),
                Gf.Vec3f(*(max(v[k] for v in vertices) for k in range(3))),
            ]
        )
        finish_shape(mesh, mats["moving" if moving else "fixed"])
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        return root

    make_panel(fixed_path, False)
    moving_prim = make_panel(moving_path, True)

    # Anchor the passive hinge to the world; the blue triangle is static.
    # USD joint angles are degrees.
    hinge = UsdPhysics.RevoluteJoint.Define(stage, board_root + "/DiagonalHinge")
    hinge.CreateBody1Rel().SetTargets([Sdf.Path(moving_path)])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*board.hinge_world))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(*board.hinge_local))
    joint_rotation = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    hinge.CreateLocalRot0Attr(usd_quat(joint_rotation))
    hinge.CreateLocalRot1Attr(usd_quat(joint_rotation))
    hinge.CreateAxisAttr("X")  # Local X rotated +45 degrees = square's diagonal.
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(180.0)
    hinge.CreateCollisionEnabledAttr(True)

    # A decorative hinge pin. The revolute constraint above supplies its physics.
    pin = UsdGeom.Cylinder.Define(stage, fixed_path + "/HingePin")
    pin.CreateAxisAttr("X")
    pin.CreateRadiusAttr(0.003)
    pin.CreateHeightAttr(board.side * math.sqrt(2) - 0.015)
    transform(pin.GetPrim(), board.hinge_local, joint_rotation)
    finish_shape(pin, mats["metal"], collision=False)

    # Small outboard handle: it keeps the gripper clear of the stacked triangles.
    # Both shapes belong to the moving panel's single rigid body.
    corner = (-board.side / 2, board.side / 2, board.thickness / 2)
    handle = board.handle_local
    stem = UsdGeom.Cylinder.Define(stage, moving_path + "/HandleStem")
    stem.CreateAxisAttr("X")
    stem.CreateHeightAttr(board.handle_extension + 0.008)
    stem.CreateRadiusAttr(0.003)
    yaw = 3 * math.pi / 4
    transform(stem.GetPrim(), scale(add(corner, handle), 0.5), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    finish_shape(stem, mats["metal"])
    sphere = UsdGeom.Sphere.Define(stage, moving_path + "/GraspPoint")
    sphere.CreateRadiusAttr(board.handle_radius)
    transform(sphere.GetPrim(), handle)
    finish_shape(sphere, mats["metal"])

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, board.table_top + 0.001)
    robot = Articulation(robot_cfg)
    panel = RigidObject(
        RigidObjectCfg(prim_path=moving_path, spawn=None, init_state=RigidObjectCfg.InitialStateCfg(pos=board.origin))
    )

    def rigid_path(name):
        matches = [
            p.GetPath()
            for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot"))
            if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Franka rigid body {name!r}; found {matches}.")
        return matches[0]

    hand_path = rigid_path("panda_hand")
    # The grasp is intentionally idealized, so finger friction is not simulated.
    # Keep table/robot and fixed-panel/robot collisions; filter only the moving
    # body against the hand/fingers to avoid double-constraining the ideal pinch.
    filtered = UsdPhysics.FilteredPairsAPI.Apply(moving_prim).CreateFilteredPairsRel()
    for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        filtered.AddTarget(rigid_path(name))

    pinch = UsdPhysics.SphericalJoint.Define(stage, "/World/IdealPinch")
    pinch.CreateBody0Rel().SetTargets([hand_path])
    pinch.CreateBody1Rel().SetTargets([Sdf.Path(moving_path)])
    pinch.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.1034))
    pinch.CreateLocalPos1Attr(Gf.Vec3f(*board.handle_local))
    pinch.CreateLocalRot0Attr(Gf.Quatf(1.0))
    pinch.CreateLocalRot1Attr(Gf.Quatf(1.0))
    pinch.CreateExcludeFromArticulationAttr(True)
    pinch.CreateJointEnabledAttr(False)

    # Publish the manually authored USD scene to non-Kit visualizers such as Viser.
    with ReplicateSession([AssetBaseCfg(prim_path="/World")], num_clones=1, env_spacing=0.0):
        pass

    sim.reset()
    q0 = robot.data.default_joint_pos.torch.clone()
    v0 = torch.zeros_like(q0)
    robot.write_joint_position_to_sim_index(position=q0)
    robot.write_joint_velocity_to_sim_index(velocity=v0)
    robot.reset()
    panel.write_root_pose_to_sim_index(root_pose=panel.data.default_root_pose.torch)
    panel.reset()
    sim.step()
    robot.update(dt)
    panel.update(dt)

    arm_ids, _ = robot.find_joints("panda_joint[1-7]")
    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    hand_ids, _ = robot.find_bodies("panda_hand")
    hand_id = hand_ids[0]
    jacobian_body = hand_id - 1 if robot.is_fixed_base else hand_id
    jacobian_ids = [j + robot.num_base_dofs for j in arm_ids]
    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=sim.device,
    )
    # The TCP is the midpoint between the fingertips, offset from panda_hand.
    tcp_offset = torch.tensor([[0.0, 0.0, 0.1034]], device=sim.device)
    # Downward tool axis, finger-closing axis parallel to the diagonal hinge.
    down_q = (math.cos(-math.pi / 8), math.sin(-math.pi / 8), 0.0, 0.0)
    limits = robot.data.soft_joint_pos_limits.torch[:, arm_ids, :]

    def targets(value, ids):
        robot.actuators.target_command.set_position_index(value=value, joint_ids=ids)

    def hand_pose():
        return robot.data.body_pose_w.torch[:, hand_id, :]

    def tcp_pose():
        hand = hand_pose()
        pos = hand[:, :3] + quat_apply(hand[:, 3:7], tcp_offset)
        return pos, hand[:, 3:7]

    def tcp_python():
        p, q = tcp_pose()
        return tuple(p[0].tolist()), tuple(q[0].tolist())

    def panel_pose():
        pose = panel.data.root_pose_w.torch[0].tolist()
        return tuple(pose[:3]), tuple(pose[3:7])

    def measured_handle():
        p, q = panel_pose()
        return add(p, rotate(q, board.handle_local))

    def measured_angle():
        return hinge_angle(panel_pose()[1])

    def control(goal_p, goal_q, opening):
        hand = hand_pose()
        root = robot.data.root_pose_w.torch
        tcp_p, tcp_q = tcp_pose()
        q = robot.data.joint_pos.torch[:, arm_ids]
        full_j = robot.data.body_link_jacobian_w.torch
        j = full_j[:, jacobian_body, :, :][:, :, jacobian_ids].clone()
        r = quat_apply(hand[:, 3:7], tcp_offset)
        # Shift the linear Jacobian from the hand origin to the fingertip TCP.
        j[:, :3, :] += torch.cross(j[:, 3:, :].transpose(1, 2), r[:, None, :].expand(-1, 7, -1), dim=-1).transpose(1, 2)
        # IK poses and both Jacobian blocks must use the same robot-root frame.
        base_q = root[:, None, 3:7].expand(-1, 7, -1)
        j[:, :3, :] = quat_apply_inverse(base_q, j[:, :3, :].transpose(1, 2)).transpose(1, 2)
        j[:, 3:, :] = quat_apply_inverse(base_q, j[:, 3:, :].transpose(1, 2)).transpose(1, 2)
        current_p, current_q = subtract_frame_transforms(root[:, :3], root[:, 3:7], tcp_p, tcp_q)
        goal = torch.tensor([[*goal_p, *goal_q]], device=sim.device, dtype=torch.float32)
        target_p, target_q = subtract_frame_transforms(root[:, :3], root[:, 3:7], goal[:, :3], goal[:, 3:7])
        ik.set_command(torch.cat((target_p, target_q), dim=-1))
        q_des = ik.compute(current_p, current_q, j, q)
        if not bool(torch.isfinite(q_des).all()):
            raise RuntimeError("IK produced non-finite joint targets.")
        q_des = q + (q_des - q).clamp(-0.12, 0.12)
        q_des = torch.maximum(torch.minimum(q_des, limits[..., 1]), limits[..., 0])
        targets(q_des, arm_ids)
        targets(torch.full((1, 2), opening, device=sim.device), finger_ids)
        robot.write_data_to_sim()

    steps = 0

    def step(goal_p, goal_q, opening):
        nonlocal steps
        if not app.is_running():
            raise KeyboardInterrupt()
        control(goal_p, goal_q, opening)
        sim.step(render=steps % 4 == 0)
        robot.update(dt)
        panel.update(dt)
        steps += 1
        return norm(sub(tcp_python()[0], goal_p))

    def segment(end_p, end_q, seconds, start_open, end_open):
        start_p, start_q = tcp_python()
        n = max(1, math.ceil(seconds / dt))
        for k in range(1, n + 1):
            u = smoothstep(k / n)
            p = add(scale(start_p, 1 - u), scale(end_p, u))
            q = slerp(start_q, end_q, u)
            step(p, q, (1 - u) * start_open + u * end_open)

    def hold(p, q, seconds, opening):
        for _ in range(math.ceil(seconds / dt)):
            step(p, q, opening)

    p, q = tcp_python()
    hold(p, q, 0.8, 0.04)
    pick = measured_handle()
    pregrasp = add(pick, (0.0, 0.0, 0.16))
    segment(pregrasp, down_q, 3.0, 0.04, 0.04)
    segment(pick, down_q, 3.0, 0.04, 0.04)
    hold(pick, down_q, 1.5, 0.04)
    tip, orientation = tcp_python()
    angle_error = 2 * math.acos(min(1.0, abs(dot(orientation, down_q))))
    distance = norm(sub(tip, measured_handle()))
    if distance > 0.012 or angle_error > 0.15:
        raise RuntimeError(
            f"Grasp alignment failed: distance={distance:.4f} m, orientation={angle_error:.3f} rad. "
            "No attachment was made. Check reachability or reduce --board-size."
        )
    segment(pick, down_q, 0.8, 0.04, board.handle_radius)
    # Establish a point constraint at the ACTUAL contact point, avoiding a
    # teleport or impulse caused by an offset between the two joint frames.
    hand = hand_pose()[0].tolist()
    local_pinch = inverse_rotate(hand[3:7], sub(measured_handle(), hand[:3]))
    if norm(sub(tcp_python()[0], measured_handle())) > 0.015:
        raise RuntimeError("The handle moved out of reach while closing the fingers.")
    pinch.GetLocalPos0Attr().Set(Gf.Vec3f(*local_pinch))
    tcp_offset[:] = torch.tensor([local_pinch], device=sim.device)
    pinch.GetJointEnabledAttr().Set(True)

    # Keep the grasp attached and reverse at each endpoint.
    end_angle = math.pi
    while app.is_running():
        move_panel(board, end_angle, args.fold_seconds, dt, measured_angle, step, hold, down_q)
        end_angle = 0.0 if end_angle == math.pi else math.pi


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=float, default=0.30, help="Square side in metres.")
    parser.add_argument("--fold-seconds", type=float, default=12.0, help="Duration of each fold or unfold.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    if not math.isfinite(args.fold_seconds) or args.fold_seconds < 4:
        parser.error("--fold-seconds must be finite and at least 4.")
    launcher = AppLauncher(vars(args).copy())
    exit_code = 0
    try:
        run(args, launcher.app)
    except KeyboardInterrupt:
        print("[INFO] Demo stopped.")
    except Exception:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
