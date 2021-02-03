import ScriptEnv

def hfss_message(proj, design, msg, priority):
    """
    Displays a string in HFSS; relies on project name and design name in global variables
    :param msg: string to display in HFSS Message Manager
    :param priority: integer in [0,3]; 0 is lowest priority
    :return: none
    """
    #displays a message in the HFSS message manager
    #priority = [0,3] where 0 is lowest priority
    oDesktop.AddMessage(proj,design,priority,msg)
    return

ScriptEnv.Initialize("Ansoft.ElectronicsDesktop") #repeated initializations are unnecessary/possibly harmful
oProject = oDesktop.GetActiveProject()
project_name = oProject.GetName()
oDesign = oProject.GetActiveDesign()
design_name = oDesign.GetName()

hfss_message(project_name, design_name, "Hello World!", 0)