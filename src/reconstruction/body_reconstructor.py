"""
3D Body Reconstruction Module (Tasks 14–16).

Creates an approximate 3D human body mesh from pose landmarks and
segmentation data. Uses a simplified cylinder/ellipsoid body model
approach as the MVP, without requiring SMPL/SMPL-X model files.

For the MVP, we build a geometric body model from pose landmarks:
- Each body segment (torso, arms, legs) is approximated as a cylinder/ellipsoid
- The cylinders are positioned using the pose landmark coordinates
- Body proportions from the feature extractor inform segment widths
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    logger.warning("trimesh not available. 3D reconstruction will be limited.")


# Body segment definitions: (start_landmark, end_landmark, radius_multiplier)
BODY_SEGMENTS = {
    "torso_left": ("left_shoulder", "left_hip", 0.12),
    "torso_right": ("right_shoulder", "right_hip", 0.12),
    "torso_front": ("left_shoulder", "right_hip", 0.08),
    "torso_back": ("right_shoulder", "left_hip", 0.08),
    "upper_arm_left": ("left_shoulder", "left_elbow", 0.045),
    "upper_arm_right": ("right_shoulder", "right_elbow", 0.045),
    "forearm_left": ("left_elbow", "left_wrist", 0.035),
    "forearm_right": ("right_elbow", "right_wrist", 0.035),
    "upper_leg_left": ("left_hip", "left_knee", 0.065),
    "upper_leg_right": ("right_hip", "right_knee", 0.065),
    "lower_leg_left": ("left_knee", "left_ankle", 0.05),
    "lower_leg_right": ("right_knee", "right_ankle", 0.05),
    "neck": ("nose", "left_shoulder", 0.035),
}

# Joint sphere definitions: (landmark, radius_multiplier)
JOINT_SPHERES = {
    "head": ("nose", 0.08),
    "left_shoulder_joint": ("left_shoulder", 0.04),
    "right_shoulder_joint": ("right_shoulder", 0.04),
    "left_elbow_joint": ("left_elbow", 0.03),
    "right_elbow_joint": ("right_elbow", 0.03),
    "left_hip_joint": ("left_hip", 0.05),
    "right_hip_joint": ("right_hip", 0.05),
    "left_knee_joint": ("left_knee", 0.04),
    "right_knee_joint": ("right_knee", 0.04),
}


class BodyReconstructor:
    """
    Creates approximate 3D body meshes from pose landmarks.

    Uses a geometric approach: body segments are modeled as tapered cylinders
    connected by spherical joints, positioned according to pose landmarks.
    """

    def __init__(self, body_height_m: float = 1.70):
        """
        Args:
            body_height_m: Assumed body height in meters for scale reference.
        """
        self.body_height_m = body_height_m
        self._mesh: Optional[trimesh.Trimesh] = None
        logger.info(f"BodyReconstructor initialized: height={body_height_m}m")

    def _landmarks_to_3d(
        self, landmarks: Dict, body_height_px: float
    ) -> Dict[str, np.ndarray]:
        """
        Convert 2D landmark pixel coordinates to approximate 3D coordinates.

        Uses the pixel body height to establish a scale factor, then maps
        pixel coordinates to meters. The z-coordinate from MediaPipe is used
        for depth, scaled proportionally.
        """
        if body_height_px <= 0:
            body_height_px = 400  # fallback

        scale = self.body_height_m / body_height_px
        coords_3d = {}

        # Use nose as origin reference
        nose = landmarks.get("nose", {})
        ref_x = nose.get("pixel_x", 0)
        ref_y = nose.get("pixel_y", 0)

        for name, lm in landmarks.items():
            px = lm.get("pixel_x", 0)
            py = lm.get("pixel_y", 0)
            z_norm = lm.get("z", 0)

            # Convert to meters relative to nose
            x = (px - ref_x) * scale
            y = -(py - ref_y) * scale  # Flip Y so up is positive
            z = z_norm * self.body_height_m * 0.3  # z is relative depth

            coords_3d[name] = np.array([x, y, z])

        return coords_3d

    def _create_cylinder(
        self, start: np.ndarray, end: np.ndarray, radius: float, segments: int = 12
    ) -> "trimesh.Trimesh":
        """Create a cylinder mesh between two 3D points."""
        direction = end - start
        height = np.linalg.norm(direction)

        if height < 1e-6:
            return trimesh.primitives.Sphere(radius=radius, center=start).to_mesh()

        # Create cylinder along Z axis then rotate to align with direction
        cylinder = trimesh.creation.cylinder(
            radius=radius, height=height, sections=segments
        )

        # Align cylinder: default is along Z, we need to align with direction
        direction_norm = direction / height
        z_axis = np.array([0, 0, 1])

        # Rotation from z-axis to direction
        v = np.cross(z_axis, direction_norm)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction_norm)

        if s < 1e-6:
            if c > 0:
                rotation = np.eye(4)
            else:
                rotation = np.eye(4)
                rotation[2, 2] = -1
                rotation[1, 1] = -1
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ])
            rot = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
            rotation = np.eye(4)
            rotation[:3, :3] = rot

        midpoint = (start + end) / 2
        rotation[:3, 3] = midpoint

        cylinder.apply_transform(rotation)
        return cylinder

    def reconstruct(
        self,
        landmarks: Dict,
        features: Optional[Dict] = None,
    ) -> Optional["trimesh.Trimesh"]:
        """
        Reconstruct a 3D body mesh from landmarks and features.

        Args:
            landmarks: Dict of pose landmarks with pixel_x, pixel_y, z.
            features: Optional body features dict for proportions.

        Returns:
            Combined trimesh.Trimesh object, or None if trimesh unavailable.
        """
        if not HAS_TRIMESH:
            logger.error("trimesh is required for 3D reconstruction")
            return None

        # Determine body height in pixels
        body_height_px = 400  # default
        if features and "body_height_px" in features:
            body_height_px = features["body_height_px"]

        # Convert landmarks to 3D coordinates
        coords_3d = self._landmarks_to_3d(landmarks, body_height_px)

        meshes = []

        # Create body segments (cylinders)
        for seg_name, (start_lm, end_lm, radius_mult) in BODY_SEGMENTS.items():
            if start_lm in coords_3d and end_lm in coords_3d:
                start = coords_3d[start_lm]
                end = coords_3d[end_lm]
                radius = self.body_height_m * radius_mult

                try:
                    cyl = self._create_cylinder(start, end, radius)
                    meshes.append(cyl)
                except Exception as e:
                    logger.debug(f"Failed to create segment {seg_name}: {e}")

        # Create joint spheres
        for joint_name, (lm_name, radius_mult) in JOINT_SPHERES.items():
            if lm_name in coords_3d:
                center = coords_3d[lm_name]
                radius = self.body_height_m * radius_mult
                try:
                    sphere = trimesh.primitives.Sphere(
                        radius=radius, center=center
                    ).to_mesh()
                    meshes.append(sphere)
                except Exception as e:
                    logger.debug(f"Failed to create joint {joint_name}: {e}")

        # Create torso fill (ellipsoid between shoulders and hips)
        if all(k in coords_3d for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            try:
                torso_center = (
                    coords_3d["left_shoulder"] + coords_3d["right_shoulder"]
                    + coords_3d["left_hip"] + coords_3d["right_hip"]
                ) / 4

                shoulder_w = np.linalg.norm(coords_3d["left_shoulder"] - coords_3d["right_shoulder"])
                torso_h = np.linalg.norm(
                    (coords_3d["left_shoulder"] + coords_3d["right_shoulder"]) / 2
                    - (coords_3d["left_hip"] + coords_3d["right_hip"]) / 2
                )
                depth = shoulder_w * 0.5  # approximate depth

                torso_ellipsoid = trimesh.creation.icosphere(subdivisions=2)
                torso_ellipsoid.apply_scale([shoulder_w / 2, torso_h / 2, depth / 2])
                torso_ellipsoid.apply_translation(torso_center)
                meshes.append(torso_ellipsoid)
            except Exception as e:
                logger.debug(f"Failed to create torso fill: {e}")

        if not meshes:
            logger.error("No body segments could be created")
            return None

        # Combine all meshes
        combined = trimesh.util.concatenate(meshes)

        # Apply a skin-like color
        combined.visual.face_colors = [220, 185, 155, 255]  # skin tone

        self._mesh = combined
        logger.info(
            f"3D reconstruction complete: {len(combined.vertices)} vertices, "
            f"{len(combined.faces)} faces"
        )
        return combined

    def render_views(
        self, output_dir: str = "outputs/images"
    ) -> List[str]:
        """
        Render the 3D mesh from multiple angles and save as images.

        Returns:
            List of saved image paths.
        """
        if self._mesh is None:
            logger.warning("No mesh available to render")
            return []

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        # Define camera angles: (name, rotation_angles_degrees)
        views = {
            "front": (0, 0),
            "side": (0, 90),
            "back": (0, 180),
            "45deg": (0, 45),
        }

        try:
            scene = self._mesh.scene()

            for view_name, (elev, azim) in views.items():
                # Create rotation matrix
                elev_rad = math.radians(elev)
                azim_rad = math.radians(azim)

                rot_y = trimesh.transformations.rotation_matrix(azim_rad, [0, 1, 0])
                rot_x = trimesh.transformations.rotation_matrix(elev_rad, [1, 0, 0])
                transform = rot_x @ rot_y

                # Apply view transform
                rotated = self._mesh.copy()
                rotated.apply_transform(transform)

                # Render to image
                rotated_scene = rotated.scene()
                try:
                    png_data = rotated_scene.save_image(resolution=(800, 1000))
                    if png_data is not None:
                        img_path = str(out / f"render_{view_name}.png")
                        with open(img_path, "wb") as f:
                            f.write(png_data)
                        saved.append(img_path)
                        logger.info(f"Render saved: {img_path}")
                except Exception as e:
                    logger.warning(f"Could not render {view_name} view: {e}")

        except Exception as e:
            logger.error(f"Rendering failed: {e}")

        return saved

    def save_mesh(
        self,
        output_dir: str = "outputs/meshes",
        formats: List[str] = None,
    ) -> List[str]:
        """
        Export the mesh in multiple formats.

        Args:
            output_dir: Directory to save mesh files.
            formats: List of format extensions (default: ["obj", "glb", "ply"]).

        Returns:
            List of saved file paths.
        """
        if self._mesh is None:
            logger.warning("No mesh to save")
            return []

        if formats is None:
            formats = ["obj", "glb", "ply"]

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        for fmt in formats:
            file_path = str(out / f"body_model.{fmt}")
            try:
                self._mesh.export(file_path)
                saved.append(file_path)
                logger.info(f"Mesh exported: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to export {fmt}: {e}")

        return saved

    def get_mesh(self) -> Optional["trimesh.Trimesh"]:
        """Return the current mesh object."""
        return self._mesh
