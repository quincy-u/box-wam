import sapien.core as sapien
import numpy as np
from pathlib import Path
import transforms3d as t3d
import sapien.physx as sapienp
import json
import os, re
import xml.etree.ElementTree as ET

from .actor_utils import Actor, ArticulationActor


def _rpy_to_quat(rpy):
    """Convert URDF rpy (roll, pitch, yaw) to [qw, qx, qy, qz]."""
    roll, pitch, yaw = rpy
    R = t3d.euler.euler2mat(roll, pitch, yaw, axes='sxyz')
    q = t3d.quaternions.mat2quat(R)
    return q.tolist()


def _get_body_rotation_from_urdf(urdf_path):
    """Return [qw,qx,qy,qz] rotation from the URDF root/base link to the
    first visual body link via fixed joints, or None if no fixed joint exists."""
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        for joint in root.findall('joint'):
            if joint.get('type') == 'fixed':
                parent_el = joint.find('parent')
                origin_el = joint.find('origin')
                if parent_el is not None and parent_el.get('link') == 'base':
                    rpy_str = origin_el.get('rpy', '0 0 0') if origin_el is not None else '0 0 0'
                    rpy = [float(v) for v in rpy_str.split()]
                    if any(abs(v) > 1e-6 for v in rpy):
                        return _rpy_to_quat(rpy)
    except Exception:
        pass
    return None


def _parse_obj_vertices(filepath):
    """Parse vertex positions from an OBJ file. Returns (N,3) array or None."""
    verts = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception:
        pass
    return np.array(verts) if verts else None


def _compute_link_bboxes_from_urdf(model_dir: Path, urdf_path: Path) -> dict:
    """Compute per-link bounding boxes (in each link's own frame, unscaled)
    from the mesh OBJ files referenced in the URDF.

    Returns dict mapping link_name -> {"min": [...], "max": [...],
                                        "center": [...], "extents": [...]}
    Only links with visual geometry are included.
    """
    cache_path = model_dir / "link_bboxes_cache.json"
    if cache_path.exists():
        return json.load(open(cache_path))

    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
    except Exception:
        return {}

    result = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        all_verts = []
        for visual in link.findall('visual'):
            origin_el = visual.find('origin')
            xyz = [0.0, 0.0, 0.0]
            if origin_el is not None:
                xyz = [float(v) for v in origin_el.get('xyz', '0 0 0').split()]
            geom = visual.find('geometry/mesh')
            if geom is not None:
                mesh_path = model_dir / geom.get('filename')
                verts = _parse_obj_vertices(mesh_path)
                if verts is not None and len(verts) > 0:
                    all_verts.append(verts + np.array(xyz))
        if not all_verts:
            continue
        combined = np.concatenate(all_verts, axis=0)
        mn = combined.min(axis=0).tolist()
        mx = combined.max(axis=0).tolist()
        result[link_name] = {
            "min": mn, "max": mx,
            "center": [(mn[i] + mx[i]) / 2 for i in range(3)],
            "extents": [mx[i] - mn[i] for i in range(3)],
        }

    if result:
        with open(cache_path, 'w') as f:
            json.dump(result, f)
    return result


def _get_revolute_joints_from_urdf(urdf_path: Path) -> list:
    """Parse revolute joints from URDF. Returns list of dicts:
       {"name", "parent", "child", "origin_xyz", "axis"}
    """
    joints = []
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        for joint in root.findall('joint'):
            if joint.get('type') == 'revolute':
                parent_el = joint.find('parent')
                child_el = joint.find('child')
                origin_el = joint.find('origin')
                axis_el = joint.find('axis')
                xyz = [0.0, 0.0, 0.0]
                if origin_el is not None:
                    xyz = [float(v) for v in origin_el.get('xyz', '0 0 0').split()]
                axis = [0.0, 0.0, 1.0]
                if axis_el is not None:
                    axis = [float(v) for v in axis_el.get('xyz', '0 0 1').split()]
                joints.append({
                    "name": joint.get('name'),
                    "parent": parent_el.get('link') if parent_el is not None else None,
                    "child": child_el.get('link') if child_el is not None else None,
                    "origin_xyz": xyz,
                    "axis": axis,
                })
    except Exception:
        pass
    return joints


class UnStableError(Exception):

    def __init__(self, msg):
        super().__init__(msg)


def preprocess(scene, pose: sapien.Pose) -> tuple[sapien.Scene, sapien.Pose]:
    """Add entity to scene. Add bias to z axis if scene is not sapien.Scene."""
    if isinstance(scene, sapien.Scene):
        return scene, pose
    else:
        return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)


# create box
def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    boxtype="default",
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
    )
    if boxtype == "default":
        # `actor_utils.get_point` uses `config["scale"]` to map unit-cube
        # contact points into half-size space, so `scale = half_size` is load-
        # bearing for grasp planning — DO NOT change it. The `_synthetic_box`
        # marker tells `Base_Task._box_corners_world` to interpret `extents`
        # as already-scaled half-size and skip the usual `extents*scale/2`
        # formula (which would otherwise produce a microscopic OBB).
        data = {
            "_synthetic_box": True,
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        # Same `_synthetic_box` marker as the default branch — without it
        # `_box_corners_world` would compute `extents * scale / 2 = hs² / 2`
        # (microscopic). The marker tells the OBB consumer to use `extents`
        # directly as the half-size.
        data = {
            "_synthetic_box": True,
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data)


# create spere
def create_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeSphere(radius=radius, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/textures/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeSphere(radius=radius, material=material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create cylinder
def create_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=scene.default_physical_material,
        ))

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=sapien.render.RenderMaterial(base_color=[*color[:3], 1]),
        ))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create box
def create_visual_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])))

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_table(
        scene,
        pose: sapien.Pose,
        length: float,
        width: float,
        height: float,
        thickness=0.1,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    # Tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])  # Center the tabletop at z=0
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(
        pose=tabletop_pose,
        half_size=tabletop_half_size,
        material=scene.default_physical_material,
    )

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
    else:
        builder.add_box_visual(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=color,
        )

    # Table legs (x4)
    leg_spacing = 0.1
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_spacing / 2)
            y = j * (width / 2 - leg_spacing / 2)
            table_leg_pose = sapien.Pose([x, y, -height / 2 - 0.002])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2 - 0.002]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    return table


# create obj model
def create_obj(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
        no_collision=False,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "textured.obj"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"textured{model_id}.obj"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if not no_collision:
        if convex == True:
            builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
        else:
            builder.add_nonconvex_collision_from_file(filename=str(file_name), scale=scale)

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    a = Actor(mesh, model_data)
    a.modelname, a.model_id = modelname, model_id
    return a


# create glb model
def create_glb(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "base.glb"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"base{model_id}.glb"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(file_name),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    a = Actor(mesh, model_data)
    a.modelname, a.model_id = modelname, model_id
    return a


def get_glb_or_obj_file(modeldir, model_id):
    modeldir = Path(modeldir)
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        if model_id is None:
            file = modeldir / "textured.obj"
        else:
            file = modeldir / f"textured{model_id}.obj"
    return file


def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=0,
) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets/objects") / modelname

    if model_id is None:
        json_file_path = modeldir / "model_data.json"
    else:
        json_file_path = modeldir / f"model_data{model_id}.json"

    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)

    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)

    if not collision_file.exists() or not visual_file.exists():
        print(modelname, "is not exist model file!")
        return None

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_name(modelname)
    mesh.set_pose(pose)
    a = Actor(mesh, model_data)
    a.modelname, a.model_id = modelname, model_id
    return a


# create urdf model
def create_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    json_file_path = modeldir / "model_data.json"
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        loader.scale = model_data["scale"][0]
    except:
        model_data = None

    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object: sapien.Articulation = loader.load(str(modeldir / "mobility.urdf"))

    object.set_root_pose(pose)
    object.set_name(modelname)
    a = ArticulationActor(object, model_data)
    a.modelname = modelname
    return a


def create_sapien_urdf_obj(
    scene,
    pose: sapien.Pose,
    modelname: str,
    scale=1.0,
    modelid: int = None,
    fix_root_link=False,
) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets") / "objects" / modelname
    if modelid is not None:
        model_list = [model for model in modeldir.iterdir() if model.is_dir() and model.name != "visual"]

        def extract_number(filename):
            match = re.search(r"\d+", filename.name)
            return int(match.group()) if match else 0

        model_list = sorted(model_list, key=extract_number)

        if modelid >= len(model_list):
            is_find = False
            for model in model_list:
                if modelid == int(model.name):
                    modeldir = model
                    is_find = True
                    break
            if not is_find:
                raise ValueError(f"modelid {modelid} is out of range for {modelname}.")
        else:
            modeldir = model_list[modelid]
    json_file = modeldir / "model_data.json"

    if json_file.exists():
        with open(json_file, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
        trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
    else:
        model_data = None
        trans_mat = np.eye(4)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object = loader.load_multiple(str(modeldir / "mobility.urdf"))[0][0]

    pose_mat = pose.to_transformation_matrix()
    pose = sapien.Pose(
        p=pose_mat[:3, 3] + trans_mat[:3, 3],
        q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
    )
    object.set_pose(pose)

    if model_data is not None:
        if "init_qpos" in model_data and len(model_data["init_qpos"]) > 0:
            object.set_qpos(np.array(model_data["init_qpos"]))
        if "mass" in model_data and len(model_data["mass"]) > 0:
            for link in object.get_links():
                link.set_mass(model_data["mass"].get(link.get_name(), 0.1))

        bounding_box_file = modeldir / "bounding_box.json"
        if bounding_box_file.exists():
            bounding_box = json.load(open(bounding_box_file, "r", encoding="utf-8"))
            bb_min = np.array(bounding_box["min"])
            bb_max = np.array(bounding_box["max"])
            model_data["extents"] = (bb_max - bb_min).tolist()
            model_data["center"] = ((bb_min + bb_max) / 2).tolist()

        # Compute rotation from articulation root (base link) to the first visual
        # body link via fixed joints. get_pose() returns the base link pose, but
        # the bounding box is measured in the body link frame, so we need this
        # offset rotation to project the OBB correctly.
        urdf_path = modeldir / "mobility.urdf"
        body_q = _get_body_rotation_from_urdf(urdf_path)
        if body_q is not None:
            model_data["body_q"] = body_q

        # Per-link bboxes (for dynamic OBB on articulated objects with revolute joints)
        link_bboxes = _compute_link_bboxes_from_urdf(modeldir, urdf_path)
        if link_bboxes:
            model_data["link_bboxes"] = link_bboxes

        revolute_joints = _get_revolute_joints_from_urdf(urdf_path)
        if revolute_joints:
            model_data["revolute_joints"] = revolute_joints
    object.set_name(modelname)
    a = ArticulationActor(object, model_data)
    a.modelname = modelname
    # `modelid` here is an index into the sorted subdir list; if the actual
    # subdir name is needed (e.g. "9748"), parse modeldir.name
    a.model_id = modeldir.name if modelid is not None else None
    return a
