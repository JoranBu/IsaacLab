# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Hanging
#
# This simulation demonstrates volumetric soft bodies (tetrahedral grids) hanging
# from fixed particles on the left side. Four grids with different damping values
# (1e4 to 1e1) showcase the effect of damping on Neo-Hookean elastic behavior.
#
# Command: uv run -m newton.examples softbody.example_softbody_hanging
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.viewer


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type != "vbd":
            raise ValueError("The hanging softbody example only supports the VBD solver.")

        builder = newton.ModelBuilder()
        # builder.add_ground_plane()

        # Grid dimensions
        dim_x = 12
        dim_y = 4
        dim_z = 4
        cell_size = 0.1

        # Create 4 grids with different damping values
        damping_values = [1e4, 1e3, 1e2, 1e1]
        spacing = 0.6  # Space between grids along Y-axis

        for i, k_damp in enumerate(damping_values):
            y_offset = i * spacing
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 1.0 + y_offset, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell_size,
                cell_y=cell_size,
                cell_z=cell_size,
                density=1.0e3,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=k_damp,
                fix_left=True,
            )
        

        builder.color()

        builder.add_shape_box(
            body=-1,
            hx=0.4,
            hy=0.4,
            hz=0.5,
            xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
        )

        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(builder, world_count=5, spacing=wp.vec3(0, 0.0, 1.0))

        self.model = scene.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))

        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that particles are in a reasonable range (soft body may settle or deform)
        # We check that they haven't exploded or collapsed completely
        # 4 grids, each roughly 1.2 x 0.4 x 0.4 in size, positioned along Y-axis
        # Initial positions: Y from 1.0 to ~3.2, X from 0 to 1.2, Z around 1.0 to 1.4
        # With fix_left=True, grids hang and sag significantly towards the ground
        p_lower = wp.vec3(-1.0, -0.5, 0.0)
        p_upper = wp.vec3(3.0, 4.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver (only 'vbd' supports volumetric soft bodies)",
            type=str,
            choices=["vbd"],
            default="vbd",
        )
        parser.add_argument("--save-frame", type=str, default=None, help="Render one frame offscreen to this PNG path.")
        parser.add_argument("--frame-index", type=int, default=60, help="Number of frames to simulate before saving.")
        return parser


def save_frame(args):
    """Simulate ``args.frame_index`` frames and save one rendered frame without a display."""
    import numpy as np
    import pyglet
    from PIL import Image

    # EGL offscreen context; must be set before the GL viewer loads pyglet
    pyglet.options["headless"] = True

    viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=True)
    example = Example(viewer, args)
    for _ in range(args.frame_index):
        example.step()
    example.render()
    wp.synchronize()
    Image.fromarray(np.asarray(viewer.get_frame().numpy())).save(args.save_frame)
    print(f"Saved frame {args.frame_index} to {args.save_frame}")
    viewer.close()


if __name__ == "__main__":
    parser = Example.create_parser()
    args, _ = parser.parse_known_args()
    if args.save_frame:
        save_frame(args)
    else:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)