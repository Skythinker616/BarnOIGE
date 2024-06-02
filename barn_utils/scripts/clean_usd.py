from pxr import Usd, Sdf
from omni.usd import duplicate_prim

worlds_path = "E:/Projects/IsaacBarn/worlds" # Change this to the path of your worlds folder

for index in range(1, 300):
	usd_path = worlds_path + f"/world_{index}.usd"
	output_path = worlds_path + f"/out_world_{index}.usd"
	stage = Usd.Stage.Open(usd_path)
	stage.RemovePrim("/default/physics")
	stage.RemovePrim("/default/ground_plane")
	stage.RemovePrim("/default/sun")
	default_prim = stage.GetPrimAtPath("/default")
	stage.SetDefaultPrim(default_prim)
	for cylinder_name in default_prim.GetChildrenNames():
		for prim_name in ["/link/visual/geometry", "/link/collision/geometry"]:
			prim = default_prim.GetPrimAtPath(cylinder_name + prim_name)
			property = prim.GetRelationship("material:binding")
			new_path = property.GetTargets()[0].ReplacePrefix(Sdf.Path("/Looks"), Sdf.Path("/default/Looks"))
			property.ClearTargets(False)
			property.AddTarget(new_path)
	duplicate_prim(stage, "/Looks", "/default/Looks")
	stage.RemovePrim("/Looks")
	stage.Export(output_path)
	print(f"Exported {output_path}")
