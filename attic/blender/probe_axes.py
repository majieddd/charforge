import bpy, math, sys
from mathutils import Euler, Vector
bpy.ops.wm.open_mainfile(filepath="charforge/work/char01/rigged.blend")
rig = next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
for pb in rig.pose.bones:
    pb.rotation_mode="XYZ"; pb.rotation_euler=Euler((0,0,0)); pb.location=Vector((0,0,0))
bpy.context.view_layer.update()

def tip(name):
    pb = rig.pose.bones[name]
    return (rig.matrix_world @ pb.tail).copy()

print("BONES:", sorted(b.name for b in rig.pose.bones))
for bone, child in [("right_shoulder","right_elbow"), ("left_shoulder","left_elbow"),
                    ("right_elbow","right_wrist"), ("right_hip","right_knee")]:
    if bone not in rig.pose.bones or child not in rig.pose.bones:
        print("skip", bone, child); continue
    base = tip(child)
    print(f"\n{bone} (watching {child}) rest={tuple(round(v,3) for v in base)}")
    for ax,label in [(0,"X"),(1,"Y"),(2,"Z")]:
        pb = rig.pose.bones[bone]
        e=[0,0,0]; e[ax]=math.radians(40)
        pb.rotation_euler=Euler(e); bpy.context.view_layer.update()
        d = tip(child) - base
        print(f"   +40deg local {label}: hand moves dX={d.x:+.3f} dY={d.y:+.3f} dZ={d.z:+.3f}")
        pb.rotation_euler=Euler((0,0,0)); bpy.context.view_layer.update()
