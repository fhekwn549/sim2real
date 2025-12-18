#!/usr/bin/env python3
"""
펜 USD 생성 스크립트

펜 구조 (뚜껑 씌운 상태):
- 뒷캡: 원통, Ø13.5mm, 길이 5mm
- 본체: 원뿔대, Ø19.8mm → Ø17mm, 길이 81.7mm
- 펜촉 뚜껑: 원뿔대, Ø17mm → Ø16mm, 길이 29mm + 반구 5mm
- 전체 길이: 120.7mm
- 무게: 16.3g

사용법:
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/CoWriteBotRL/pen_grasp_rl/scripts/create_pen_usd.py
"""

import math
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

# ============================================================================
# 펜 치수 (미터 단위)
# ============================================================================
TOTAL_LENGTH = 0.1207       # 전체 길이: 120.7mm
BACK_CAP_DIAMETER = 0.0135  # 뒷캡 지름: 13.5mm
BACK_CAP_LENGTH = 0.005     # 뒷캡 길이: 5mm
MAX_DIAMETER = 0.0198       # 최대 지름: 19.8mm (뒤쪽)

# 펜촉 뚜껑 치수
CAP_LENGTH = 0.034          # 뚜껑 전체 길이: 34mm
CAP_MAX_DIAMETER = 0.017    # 뚜껑 최대 지름: 17mm
CAP_CONE_LENGTH = 0.029     # 원뿔대 부분 길이: 29mm
CAP_MIN_DIAMETER = 0.016    # 뚜껑 최소 지름: 16mm
CAP_TIP_LENGTH = 0.005      # 반구 부분 길이: 5mm

# 본체 치수 계산
BODY_LENGTH = TOTAL_LENGTH - BACK_CAP_LENGTH - CAP_LENGTH  # 본체 길이: 81.7mm
BODY_MIN_DIAMETER = CAP_MAX_DIAMETER  # 본체 최소지름 = 뚜껑 최대지름: 17mm

PEN_MASS = 0.0163           # 무게: 16.3g

# 색상 (RGB)
BLUE_COLOR = (0.0, 0.2, 0.8)   # 파란색 (뒷캡, 펜촉 뚜껑)


def create_truncated_cone_mesh(radius_top, radius_bottom, height, segments=32):
    """원뿔대(truncated cone) 메시 생성"""
    points = []
    face_vertex_counts = []
    face_vertex_indices = []

    # 위쪽 원 (z = height/2)
    top_center_idx = 0
    points.append(Gf.Vec3f(0, 0, height/2))
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius_top * math.cos(angle)
        y = radius_top * math.sin(angle)
        points.append(Gf.Vec3f(x, y, height/2))

    # 아래쪽 원 (z = -height/2)
    bottom_center_idx = len(points)
    points.append(Gf.Vec3f(0, 0, -height/2))
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius_bottom * math.cos(angle)
        y = radius_bottom * math.sin(angle)
        points.append(Gf.Vec3f(x, y, -height/2))

    # 위쪽 면
    for i in range(segments):
        face_vertex_counts.append(3)
        next_i = (i + 1) % segments
        face_vertex_indices.extend([top_center_idx, 1 + i, 1 + next_i])

    # 아래쪽 면
    for i in range(segments):
        face_vertex_counts.append(3)
        next_i = (i + 1) % segments
        face_vertex_indices.extend([bottom_center_idx, bottom_center_idx + 1 + next_i, bottom_center_idx + 1 + i])

    # 측면
    for i in range(segments):
        face_vertex_counts.append(4)
        next_i = (i + 1) % segments
        top_curr = 1 + i
        top_next = 1 + next_i
        bottom_curr = bottom_center_idx + 1 + i
        bottom_next = bottom_center_idx + 1 + next_i
        face_vertex_indices.extend([top_curr, bottom_curr, bottom_next, top_next])

    return points, face_vertex_counts, face_vertex_indices


def create_hemisphere_mesh(radius, height, segments=32, rings=8):
    """반구 메시 생성 (아래쪽으로 볼록)"""
    points = []
    face_vertex_counts = []
    face_vertex_indices = []

    # 상단 원 (z = 0, 본체와 연결되는 부분)
    top_center_idx = 0
    points.append(Gf.Vec3f(0, 0, 0))
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append(Gf.Vec3f(x, y, 0))

    # 반구 표면 점들
    for ring in range(1, rings):
        phi = (math.pi / 2) * ring / rings  # 0 ~ 90도
        r = radius * math.cos(phi)
        z = -radius * math.sin(phi)  # 아래쪽으로
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append(Gf.Vec3f(x, y, z))

    # 맨 아래 점
    bottom_idx = len(points)
    points.append(Gf.Vec3f(0, 0, -radius))

    # 상단 면 (평평한 부분)
    for i in range(segments):
        face_vertex_counts.append(3)
        next_i = (i + 1) % segments
        face_vertex_indices.extend([top_center_idx, 1 + next_i, 1 + i])

    # 반구 측면
    for ring in range(rings - 1):
        for i in range(segments):
            next_i = (i + 1) % segments
            if ring == 0:
                curr_ring_start = 1
            else:
                curr_ring_start = 1 + ring * segments
            next_ring_start = 1 + (ring + 1) * segments

            face_vertex_counts.append(4)
            face_vertex_indices.extend([
                curr_ring_start + i,
                curr_ring_start + next_i,
                next_ring_start + next_i,
                next_ring_start + i
            ])

    # 맨 아래 삼각형들
    last_ring_start = 1 + (rings - 1) * segments
    for i in range(segments):
        face_vertex_counts.append(3)
        next_i = (i + 1) % segments
        face_vertex_indices.extend([last_ring_start + i, last_ring_start + next_i, bottom_idx])

    return points, face_vertex_counts, face_vertex_indices


def create_pen_usd(output_path):
    """펜 USD 파일 생성"""
    stage = Usd.Stage.CreateNew(output_path)
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)

    # 루트 Xform
    root_path = "/Pen"
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root_xform.GetPrim())

    # RigidBody API
    UsdPhysics.RigidBodyAPI.Apply(root_xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root_xform.GetPrim())
    mass_api.CreateMassAttr().Set(PEN_MASS)

    # 파란색 머티리얼
    blue_mat_path = f"{root_path}/Materials/BlueMaterial"
    blue_material = UsdShade.Material.Define(stage, blue_mat_path)
    blue_shader = UsdShade.Shader.Define(stage, f"{blue_mat_path}/Shader")
    blue_shader.CreateIdAttr("UsdPreviewSurface")
    blue_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*BLUE_COLOR))
    blue_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    blue_material.CreateSurfaceOutput().ConnectToSource(blue_shader.ConnectableAPI(), "surface")

    # ========================================================================
    # 펜 중심이 원점에 오도록 Z 오프셋 계산
    # 전체 구조: 뒷캡(5mm) + 본체(81.7mm) + 뚜껑(34mm) = 120.7mm
    # ========================================================================
    half_length = TOTAL_LENGTH / 2

    # ========================================================================
    # 1. 뒷캡 (원통) - 맨 뒤
    # ========================================================================
    cap_path = f"{root_path}/BackCap"
    cap_cylinder = UsdGeom.Cylinder.Define(stage, cap_path)
    cap_cylinder.CreateRadiusAttr().Set(BACK_CAP_DIAMETER / 2)
    cap_cylinder.CreateHeightAttr().Set(BACK_CAP_LENGTH)
    cap_cylinder.CreateAxisAttr().Set("Z")

    # 위치: 맨 뒤 (z = half_length - BACK_CAP_LENGTH/2)
    cap_z = half_length - BACK_CAP_LENGTH / 2
    UsdGeom.Xformable(cap_cylinder).AddTranslateOp().Set(Gf.Vec3d(0, 0, cap_z))
    UsdPhysics.CollisionAPI.Apply(cap_cylinder.GetPrim())
    UsdShade.MaterialBindingAPI(cap_cylinder).Bind(blue_material)

    # ========================================================================
    # 2. 본체 (원뿔대) - 중간
    # ========================================================================
    body_path = f"{root_path}/Body"
    body_mesh = UsdGeom.Mesh.Define(stage, body_path)

    points, face_counts, face_indices = create_truncated_cone_mesh(
        radius_top=MAX_DIAMETER / 2,        # 뒤쪽: 19.8mm
        radius_bottom=BODY_MIN_DIAMETER / 2,  # 앞쪽: 17mm
        height=BODY_LENGTH,
        segments=32
    )
    body_mesh.CreatePointsAttr().Set(points)
    body_mesh.CreateFaceVertexCountsAttr().Set(face_counts)
    body_mesh.CreateFaceVertexIndicesAttr().Set(face_indices)

    # 위치: 뒷캡 바로 앞
    body_z = half_length - BACK_CAP_LENGTH - BODY_LENGTH / 2
    UsdGeom.Xformable(body_mesh).AddTranslateOp().Set(Gf.Vec3d(0, 0, body_z))
    UsdPhysics.CollisionAPI.Apply(body_mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(body_mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("convexHull")
    UsdShade.MaterialBindingAPI(body_mesh).Bind(blue_material)

    # ========================================================================
    # 3. 펜촉 뚜껑 - 원뿔대 부분
    # ========================================================================
    tip_cone_path = f"{root_path}/TipCone"
    tip_cone_mesh = UsdGeom.Mesh.Define(stage, tip_cone_path)

    points2, face_counts2, face_indices2 = create_truncated_cone_mesh(
        radius_top=CAP_MAX_DIAMETER / 2,    # 뒤쪽 (본체와 연결): 17mm
        radius_bottom=CAP_MIN_DIAMETER / 2,  # 앞쪽: 16mm
        height=CAP_CONE_LENGTH,
        segments=32
    )
    tip_cone_mesh.CreatePointsAttr().Set(points2)
    tip_cone_mesh.CreateFaceVertexCountsAttr().Set(face_counts2)
    tip_cone_mesh.CreateFaceVertexIndicesAttr().Set(face_indices2)

    # 위치: 본체 바로 앞
    tip_cone_z = half_length - BACK_CAP_LENGTH - BODY_LENGTH - CAP_CONE_LENGTH / 2
    UsdGeom.Xformable(tip_cone_mesh).AddTranslateOp().Set(Gf.Vec3d(0, 0, tip_cone_z))
    UsdPhysics.CollisionAPI.Apply(tip_cone_mesh.GetPrim())
    mesh_collision2 = UsdPhysics.MeshCollisionAPI.Apply(tip_cone_mesh.GetPrim())
    mesh_collision2.CreateApproximationAttr().Set("convexHull")
    UsdShade.MaterialBindingAPI(tip_cone_mesh).Bind(blue_material)

    # ========================================================================
    # 4. 펜촉 뚜껑 - 반구 부분 (둥근 끝)
    # ========================================================================
    tip_sphere_path = f"{root_path}/TipSphere"
    tip_sphere_mesh = UsdGeom.Mesh.Define(stage, tip_sphere_path)

    # 반구 반지름 = 뚜껑 최소지름/2 = 8mm
    hemisphere_radius = CAP_MIN_DIAMETER / 2
    points3, face_counts3, face_indices3 = create_hemisphere_mesh(
        radius=hemisphere_radius,
        height=CAP_TIP_LENGTH,
        segments=32,
        rings=8
    )
    tip_sphere_mesh.CreatePointsAttr().Set(points3)
    tip_sphere_mesh.CreateFaceVertexCountsAttr().Set(face_counts3)
    tip_sphere_mesh.CreateFaceVertexIndicesAttr().Set(face_indices3)

    # 위치: 원뿔대 바로 앞 (반구 상단이 원뿔대 하단과 연결)
    tip_sphere_z = half_length - BACK_CAP_LENGTH - BODY_LENGTH - CAP_CONE_LENGTH
    UsdGeom.Xformable(tip_sphere_mesh).AddTranslateOp().Set(Gf.Vec3d(0, 0, tip_sphere_z))
    UsdPhysics.CollisionAPI.Apply(tip_sphere_mesh.GetPrim())
    mesh_collision3 = UsdPhysics.MeshCollisionAPI.Apply(tip_sphere_mesh.GetPrim())
    mesh_collision3.CreateApproximationAttr().Set("convexHull")
    UsdShade.MaterialBindingAPI(tip_sphere_mesh).Bind(blue_material)

    # 저장
    stage.GetRootLayer().Save()
    print(f"펜 USD 생성 완료: {output_path}")
    print(f"  - 전체 길이: {TOTAL_LENGTH * 1000:.1f}mm")
    print(f"  - 뒷캡: Ø{BACK_CAP_DIAMETER * 1000:.1f}mm, {BACK_CAP_LENGTH * 1000:.1f}mm")
    print(f"  - 본체: Ø{MAX_DIAMETER * 1000:.1f}mm → Ø{BODY_MIN_DIAMETER * 1000:.1f}mm, {BODY_LENGTH * 1000:.1f}mm")
    print(f"  - 펜촉 뚜껑: Ø{CAP_MAX_DIAMETER * 1000:.1f}mm → Ø{CAP_MIN_DIAMETER * 1000:.1f}mm, {CAP_CONE_LENGTH * 1000:.1f}mm + 반구 {CAP_TIP_LENGTH * 1000:.1f}mm")
    print(f"  - 무게: {PEN_MASS * 1000:.1f}g")

    return output_path


if __name__ == "__main__":
    output_path = "/home/fhekwn549/CoWriteBotRL/pen_grasp_rl/models/pen_new.usd"
    create_pen_usd(output_path)
