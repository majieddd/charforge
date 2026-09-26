"""Render a clip as a frame sequence from a fixed camera, so it can be watched as motion."""
import bpy, sys, math, os, numpy as np
from mathutils import Vector
av=sys.argv[sys.argv.index("--")+1:]
blend,clip,outdir,res = av[0],av[1],av[2],int(av[3])
yaw = float(av[4]) if len(av)>4 else 28.0
every = int(av[5]) if len(av)>5 else 1          # render every n-th frame (a 60 fps clip against a 24 fps video)
bpy.ops.wm.open_mainfile(filepath=blend)
sc=bpy.context.scene
rig=next(o for o in sc.objects if o.type=="ARMATURE")
act=bpy.data.actions.get(clip); rig.animation_data.action=act
if hasattr(rig.animation_data,"action_slot") and getattr(act,"slots",None):
    rig.animation_data.action_slot=act.slots[0]
f0,f1=(int(x) for x in act.frame_range)
# bounds over the whole clip so the camera never clips
pts=[]
for fr in range(f0,f1+1,2):
    sc.frame_set(fr); dg=bpy.context.evaluated_depsgraph_get()
    for o in sc.objects:
        if o.type=="MESH":
            ev=o.evaluated_get(dg); m=ev.to_mesh()
            V=np.empty(len(m.vertices)*3); m.vertices.foreach_get("co",V)
            mw=np.array(o.matrix_world); pts.append(V.reshape(-1,3)@mw[:3,:3].T+mw[:3,3])
            ev.to_mesh_clear()
P=np.vstack(pts); lo,hi=P.min(0),P.max(0)
ctr=Vector((float((lo[0]+hi[0])/2),float((lo[1]+hi[1])/2),float((lo[2]+hi[2])/2)))
h=float(hi[2]-lo[2]); d=h*1.9
r=math.radians(yaw)
cam_d=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cam_d); sc.collection.objects.link(cam)
sc.camera=cam; cam_d.lens=68
cam.location=ctr+Vector((math.sin(r)*d, -math.cos(r)*d, h*0.10))
cam.rotation_euler=(ctr-cam.location).to_track_quat('-Z','Y').to_euler()
w=bpy.data.worlds.new("w"); sc.world=w; w.use_nodes=True
w.node_tree.nodes["Background"].inputs[0].default_value=(0.085,0.095,0.115,1)
w.node_tree.nodes["Background"].inputs[1].default_value=1.0
key=bpy.data.objects.new("k",bpy.data.lights.new("k",'SUN')); sc.collection.objects.link(key)
key.data.energy=4.0; key.data.angle=math.radians(12); key.rotation_euler=(math.radians(56),0,math.radians(35))
rim=bpy.data.objects.new("r",bpy.data.lights.new("r",'SUN')); sc.collection.objects.link(rim)
rim.data.energy=2.2; rim.data.color=(0.62,0.74,1.0); rim.rotation_euler=(math.radians(68),0,math.radians(205))
# ground plane so the feet read against something
bpy.ops.mesh.primitive_plane_add(size=14, location=(ctr.x,ctr.y,float(lo[2])))
gp=bpy.context.active_object
gm=bpy.data.materials.new("g"); gm.use_nodes=True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=(0.16,0.18,0.22,1)
gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value=0.95
gp.data.materials.append(gm)
eng={i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
sc.render.engine="BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
sc.render.resolution_x=res; sc.render.resolution_y=res
sc.view_settings.view_transform="AgX" if "AgX" in {t.name for t in sc.view_settings.bl_rna.properties['view_transform'].enum_items} else "Standard"
os.makedirs(outdir,exist_ok=True)
n=0
for fr in range(f0,f1+1,every):
    sc.frame_set(fr); sc.render.filepath=os.path.join(outdir,f"f{n:03d}.png")
    bpy.ops.render.render(write_still=True); n+=1
print(f"[seq] {n} frames -> {outdir}")
