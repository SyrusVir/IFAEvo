
import ScriptEnv
import random
from matrix_find import matrix_find

ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")

def config_ifa(matrix, unit_width=1, unit_height=1, Cu_thickness=35):
    '''
    Configures an HFSS model assuming presence of a coordinate syste called "EvoCS"
    
    Places copper blocks of dimensions 'unit_width'mm x 'unit_height'mm x 'Cu_thickness'um mm^3... 
    at locations corresponding to indices of entries equal to 1 in 'matrix'.

    If the HFSS design variable 'Cu_thickness is not available, it is created with the value 35um
    '''
    #find indices of 1 entries of 'matrix'
    ind_1 = matrix_find(matrix,lambda x: x == 1)

    #element dimensions in mm
    #unit_width = 1
    #unit_height = 1

    unit_width_str = "{}mm".format(unit_width)
    unit_height_str = "{}mm".format(unit_height)
    Cu_thickness_str = "{}um".format(Cu_thickness)

    obj_prefix = 'Element'

    #Clear previous configuration
    try:   
        oEditor.Delete(
            [
                "NAME:Selections",
                "Selections:="		, obj_prefix + "_Group"
            ])
    except:
        pass
        
    # set working coordinate system to that of last section
    # Not in try/except because the model must be prepared with
    # the CS in the right position to ensure proper configuration
    oEditor.SetWCS(
        [
            "NAME:SetWCS Parameter",
            "Working Coordinate System:=","EvoCS",
            "RegionDepCSOk:=",False
        ])

    #create design variables
    try:   
        oDesign.ChangeProperty(
            [
                "NAME:AllTabs",
                [
                    "NAME:LocalVariableTab",
                    [
                        "NAME:PropServers", 
                        "LocalVariables"
                    ],
                    [
                        "NAME:NewProps",
                        [
                            "NAME:Cu_thickness",
                            "PropType:="		, "VariableProp",
                            "UserDef:="		, True,
                            "Value:="		, Cu_thickness_str
                        ]
                    ]
                ]
            ])
    except:
        pass

    obj_arr = []
    for n,inds in enumerate(ind_1):
        obj_name = obj_prefix + "{}".format(n)
        obj_arr.append(obj_name)
        x_pos = "{}mm".format(unit_width*inds[1])
        y_pos = "{}mm".format(unit_height*inds[0])
        
        
        oEditor.CreateBox(
            [
                "NAME:BoxParameters",
                "XPosition:="   , x_pos,
                "YPosition:="	, y_pos,
                "ZPosition:="	, "0mm",
                "XSize:="		, unit_width_str,
                "YSize:="		, unit_height_str,
                "ZSize:="		, "-Cu_thickness",
                "WhichAxis:="		, "Z"
            ], 
            [
                "NAME:Attributes",
                "Name:="		, obj_name,
                "Flags:="		, "",
                "Color:="		, "(255 128 0)", #copper orange
                "Transparency:="	, 0,
                "PartCoordinateSystem:=", "Global",
                "UDMId:="		, "",
                "MaterialValue:="	, "\"copper\"",
                "SurfaceMaterialValue:=", "\"\"",
                "SolveInside:=" , False,
                "IsMaterialEditable:="	, True,
                "UseMaterialAppearance:=", False,
            ])

    oEditor.CreateGroup(
	[
		"NAME:GroupParameter",
		"ParentGroupID:="	, "Model",
		"Parts:="		, ",".join(obj_arr),
		"SubmodelInstances:="	, "",
		"Groups:="		, ""
	])

if __name__ == "__main__":
    rows = 3
    cols = 5
    m = [[0 for i in range(cols)] for j in range(rows)]


    for i in range(4):
        ind_r = (random.randint(0,rows-1), random.randint(0,cols-1))
        m[ind_r[0]][ind_r[1]] = 1
    print(m)

    config_ifa(m)