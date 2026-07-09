import pathlib

import mujoco
from robot_descriptions import shadow_hand_mj_description as _shd

_SHADOW_DIR = pathlib.Path(_shd.MJCF_PATH).parent


def _build_scene_spec() -> "mujoco.MjSpec":
    left_xml = str(_SHADOW_DIR / "left_hand.xml")
    right_xml = str(_SHADOW_DIR / "right_hand.xml")

    spec = mujoco.MjSpec()

    spec.visual.quality.shadowsize = 4096

    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.ambient = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    spec.visual.rgba.fog = [0.07, 0.08, 0.12, 1.0]
    spec.visual.rgba.haze = [0.09, 0.10, 0.14, 1.0]
    spec.visual.map.fogstart = 2.0
    spec.visual.map.fogend = 8.0
    spec.visual.map.shadowscale = 0.4
    spec.visual.map.haze = 0.2

    key = spec.worldbody.add_light()
    key.pos = [-0.2, -0.8, 1.2]
    key.dir = [0.15, 0.65, -0.90]
    key.diffuse = [1.00, 0.90, 0.72]
    key.specular = [0.55, 0.50, 0.40]
    key.ambient = [0.0, 0.0, 0.0]
    key.castshadow = True

    fill = spec.worldbody.add_light()
    fill.pos = [0.8, 0.5, 0.7]
    fill.dir = [-0.55, -0.40, -0.55]
    fill.diffuse = [0.28, 0.34, 0.50]
    fill.specular = [0.08, 0.10, 0.16]
    fill.ambient = [0.0, 0.0, 0.0]
    fill.castshadow = False

    rim = spec.worldbody.add_light()
    rim.pos = [0.2, 0.9, 0.6]
    rim.dir = [-0.10, -0.80, -0.45]
    rim.diffuse = [0.45, 0.45, 0.65]
    rim.specular = [0.25, 0.25, 0.38]
    rim.ambient = [0.0, 0.0, 0.0]
    rim.castshadow = False

    sky = spec.worldbody.add_light()
    sky.pos = [0.2, 0.0, 2.0]
    sky.dir = [0.0, 0.05, -1.0]
    sky.diffuse = [0.10, 0.11, 0.15]
    sky.specular = [0.0, 0.0, 0.0]
    sky.ambient = [0.07, 0.08, 0.11]
    sky.castshadow = False

    left_spec = mujoco.MjSpec.from_file(left_xml)
    right_spec = mujoco.MjSpec.from_file(right_xml)

    fl = spec.worldbody.add_frame()
    fl.pos = [0.0, 0.12, 0.0]
    fl.attach_body(left_spec.body("lh_forearm"), "L_", "")

    fr = spec.worldbody.add_frame()
    fr.pos = [0.0, -0.12, 0.0]
    fr.attach_body(right_spec.body("rh_forearm"), "R_", "")

    spec.stat.center = [0.2, 0.0, 0.02]
    spec.stat.extent = 0.55
    spec.visual.global_.azimuth = 200
    spec.visual.global_.elevation = -30

    return spec


def export_web_assets(out_dir: str) -> str:
    import re
    import shutil

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    spec = _build_scene_spec()
    spec.compile()
    xml = spec.to_xml()

    xml = re.sub(r'\s+content_type="[^"]*"', "", xml)

    (out / "scene.xml").write_text(xml)

    for obj in (_SHADOW_DIR / "assets").glob("*.obj"):
        shutil.copy2(obj, out / obj.name)

    return str(out / "scene.xml")
