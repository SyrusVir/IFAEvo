# Jude Alnas
# University of Alabama AAML
# 25-12-2020
# IFA evolved design using DHBPSO algorithm


import sys
import os
# add to PATH directory containing this script
sys.path.insert(0, os.path.dirname(__file__))
import random
import ScriptEnv
import csv
import re
import time
import copy
from matrix import matrix_find, matrix_edge_finder, update_matrix_from_position, write_matrix_file



# Set up directory structure for optimization
script_dir = os.path.dirname(__file__) #get folder containing this script
data_path = os.path.join(script_dir,'Data') #top level directory containing all data
if not os.path.exists(data_path):
    os.makedirs(data_path)
opt_dir_stem = os.path.join(data_path, time.strftime("%Y_%b_%d Optimization "))
i = 0
while True:
    # iterate until a new directory name is confirmted
    s = str(i)
    if i < 10 and i > 0:
        # single-digit padding
        s = '0'+s
    opt_dir = opt_dir_stem + s
    if os.path.exists(opt_dir):
        i += 1
    else:
        break
os.makedirs(opt_dir) # make folder called "[year]_[month]_[day] Optimization [integer]"
temp_csv = os.path.join(opt_dir,'temp.csv') #path to temporary csv to contain output of itertions

#HFSS scripting object initialization
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop") #repeated initializations are unnecessary/possibly harmful
oProject = oDesktop.GetActiveProject()
project_name = oProject.GetName()
oDesign = oProject.GetActiveDesign()
design_name = oDesign.GetName()
oEditor = oDesign.SetActiveEditor("3D Modeler")
oModule = oDesign.GetModule("BoundarySetup")

class Particle:
    def __init__(self, position, velocity, cost, best_position, best_cost):
        self.position = position
        self.velocity = velocity
        self.cost = cost
        self.best_position = best_position
        self.best_cost = best_cost
        self.local_best_position = position
        self.local_best_cost = float('inf')

def main():
    #PSO parameters
    pop = 10
    Vmax = 1
    mode = 0 #0: gbest, 0.5: hybrid, 1: lbest, -1: DH-BPSO
    last_i = 0 # no proper handling for resuming iterations yet
    max_it = 30 #max iterations for individual PSO's
    max_gen = 50 # max number of times to "grow" IFA #grid size of IFA area
    
    # matrix parameters; matrix is a representation in code of the configuration of the IFA ground clearance area
    # 4 possible entries:
    # 1) '0': None; nothing is drawn at this location
    # 2) '1': Corresponds to location of conductive patch elements
    # 3) feed_sentinel: Corresponds to location of feed; use a number != 0,1 to ensure it is not deleted/overwritten
    # 4) pad_sentinel: Non-zero value used to expand the 'growth' area without excluding their locations 
    rows = 14
    cols = 85
    feed_ind = (0, 12) # indices of lumped port excitation
    seed_inds = [(1, 12)] # patch to always include just below lumped port

    # sentinel values
    feed_sentinel = 9
    global pad_sentinel
    pad_sentinel = 5 
    
    # initialize matrix representing current antenna configuration;
    # location of 1's indicate conductive patches; 9 indicates feed; otw padding
    seed_matrix = [[0 for i in range(cols)] for j in range(rows)] #initialize
    seed_matrix[feed_ind[0]][feed_ind[1]] = feed_sentinel #add location of feed
    for ind in seed_inds: #add other seed patches
        seed_matrix[ind[0]][ind[1]] = 1 

    #frame matrix - a configuration drawn from if PSO finds no better solution
    frame_matrix = [[0 for i in range(cols)] for j in range(rows)]
    
    # 12th row; from 12th column to 2nd-to-last
    frame_matrix[12][12:-1] = [1]*len(frame_matrix[12][12:-2]) 
    
    # 12th column; from 1st to 2nd-to-last ro
    frame_matrix = [list(x) for x in zip(*frame_matrix)]    #transpose
    frame_matrix[12][:-1] = [1]*len(frame_matrix[12][:-2])  
    
    frame_matrix = [list(x) for x in zip(*frame_matrix)]    #transpose
    frame_matrix[feed_ind[0]][feed_ind[1]] = feed_sentinel  # add feed
    for ind in seed_inds:                                   # add any other seed patches 
        frame_matrix[ind[0]][ind[1]] = 1

    gen_best_cost = 0
    gen_best_matrix = copy.deepcopy(seed_matrix) #copy by value instead of references
    gen_best_matrix = copy.deepcopy(frame_matrix) #copy by value instead of references

    # counts number of times a better solution is not found by the PSO
    # equivalent to number of times padding routines are called
    stagnant = 0 
    max_stagnant = 50 # max number of times padding routines can be called before terming optimization 'failure'
    #iterate over the alloted generations
    for gen in range(max_gen):
        gen_str = "Generation {}".format(gen)
        global gen_dir
        gen_dir = os.path.join(opt_dir,gen_str)
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        
        hfss_message(project_name, design_name,0, gen_str)
        
        write_matrix_file(gen_best_matrix, os.path.join(gen_dir,"gen{} initial matrix.csv".format(gen)))

        # Calling PSO
        new_cost, new_matrix = PSO(max_it ,pop ,gen_best_matrix,Vmax, mode, last_i)
        if new_cost < gen_best_cost:
            # if a better solution is found, assign and reset retry count
            gen_best_cost = new_cost
            gen_best_matrix = copy.deepcopy(new_matrix)
            stagnant = 0
        elif stagnant < max_stagnant:
            # if max retries not met, add from frame and continue optimization
            stagnant += 1
            '''
            ind_adj = matrix_edge_finder(gen_best_matrix,pad_sentinel) # get current adjacent indices
            frame_ind1 = matrix_find(frame_matrix,lambda x: x == 1) # find indices of 1's
            
            # set of indices both currently adjacent and having a 1 in the frame matrix
            frame_and_adj = [ind for ind in frame_ind1 if ind in ind_adj] 

            for ind in frame_and_adj:
                gen_best_matrix[ind[0]][ind[1]] = frame_matrix[ind[0]][ind[1]]
            '''
            
            #do padding
            ind_adj = matrix_edge_finder(gen_best_matrix) # get current adjacent indices
            for ind_pair in ind_adj:
                #replace with padding sentinel value
                gen_best_matrix[ind_pair[0]][ind_pair[1]] = pad_sentinel            
            
        else:
            # otherwise end optimization
            break
        
        oProject.Save()

    return

def particle_to_str(p):
    out = "\nPosition: " + str(p.position) + "\nVelocity: " + str(p.velocity) + "\nCost:" + str(
        p.cost) + "\nBest Position:" + str(p.best_position) + "\nBest Cost:" + str(p.best_cost) + "\n"
    return out

def PSO(max_it, pop, matrix0, Vmax, lbest_c, last_i):
    global gen_dir, pad_sentinel
    #for the given matrix, find the indices adjacent to the non-zero entries
    ind_adj = matrix_edge_finder(matrix0, pad_sentinel)
    n_vars = len(ind_adj)
    max_pos = 2 ** n_vars - 1
    Positions_dic = {} #dictionary has cost associated with each distinct particle position /JA
    
    # Intitialization
    # Initialize to worst possible solution (infinity for minimization problems)
    global_best_cost = float('inf')
    global_best_position = -1
    global_best_matrix = matrix0

    #create iteration directory
    iter_str = "Iteration {}".format(last_i)
    iter_dir = os.path.join(gen_dir,iter_str)
    if not os.path.exists(iter_dir):
        os.makedirs(iter_dir)
    
    # Initialize population members
    swarm = []
    hfss_message(project_name, design_name,0,"Starting BPSO "+iter_str)
    for i in range(pop):
        # Generate random starting locs and velocities
        start_velocity_b = []
        start_velocity_s = ''
        ones = [] #contains indices of 1 bits

        for n in range(n_vars):
            start_velocity_b.append(random.randint(0, 1))
            if start_velocity_b[n] == 1:
                ones.append(n)
        while len(ones) > Vmax: #restrict 1 bits in velocity to "Vmax" 1 bits /JA
            n = random.randint(0, len(ones) - 1)    #randomly select a 1 bit /JA
            start_velocity_b[ones[n]] = 0           #remove from bit array /JA
            ones.pop(n)                             #remove index entry /JA

        for bit in start_velocity_b: #convert bit array to string
            start_velocity_s += str(bit)

        start_position = random.randint(0, max_pos)
        start_velocity = int(start_velocity_s, 2)
        swarm.append(Particle(start_position, start_velocity, float('inf'), 0, float('inf')))

        # Evaluation
        mat_path = os.path.join(iter_dir,"Particle_{}_matrix.csv".format(swarm[i].position))
        particle_matrix = update_matrix_from_position(swarm[i].position,matrix0,ind_adj)
        write_matrix_file(particle_matrix,mat_path)
        swarm[i].cost = evaluate(swarm[i].position, matrix0, ind_adj)
        Positions_dic[swarm[i].position] = swarm[i].cost

        # Update personal best
        if swarm[i].cost < swarm[i].best_cost:
            swarm[i].best_position = swarm[i].position
            swarm[i].best_cost = swarm[i].cost
        # Update global best
        if swarm[i].best_cost < global_best_cost:
            global_best_cost = swarm[i].best_cost
            global_best_position = swarm[i].best_position
            global_best_matrix = update_matrix_from_position(swarm[i].position,matrix0,ind_adj)

    # Update local best
    for i in range(pop):
        for n in [-1,0,1]:
            ind = (i+n) % pop #note that (-1)%x = x-1 /JA
            if swarm[ind].best_cost < swarm[i].local_best_cost:
                swarm[i].local_best_cost = swarm[ind].best_cost
                swarm[i].local_best_position = swarm[ind].best_position

    '''
    #Store swarm Data
    if last_i==0:
        with open(output_loc+"/swarm_iteration_" + str(0) + ".txt", "w") as swarm_data:
            str_out = optim_param_header(max_it, pop, n_vars, Vmax, lbest_c)
            str_out += "\n\n"
            str_out += swarm_data_string(swarm, 0, lbest_c, global_best_cost, global_best_position, 0)
            swarm_data.write(str_out)
    '''

    # Main loop of pso
    temp = lbest_c #stores lbest_c user parameter in temporary variable /JA
    for j in range(last_i, max_it):
        #create iteration directory
        iter_str = "Iteration {}".format(j+1)
        iter_dir = os.path.join(gen_dir,iter_str)
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)
    
        #Notify new iteration /JA
        message = "Starting BPSO " + iter_str
        hfss_message(project_name, design_name,0, message)

        iter_start = time.time()  #iteration start time /JA

        convergence_test = 1
        best_costs = 0
        for i in range(pop):
            # Update Velocity
            if temp == -1:
                lbest_c = 1 - j / max_it #lbest_c inversely proportional to iteration number /JA

            # Combine gbest and lbest values
            mask_b = []
            for n in range(n_vars):
                if random.random() > lbest_c: #lbest_c is a probability threshold for a 1 bit /JA
                    mask_b.append(1)
                else:
                    mask_b.append(0)
            m_str = ''
            for bit in mask_b:
                m_str += str(bit)
            mask = int(m_str, 2)
            gbest = global_best_position & mask #apply mask to global position /JA
            mask = mask ^ (2 ** n_vars - 1) #invert mask /JA
            lbest = swarm[i].local_best_position & mask #apply inverted mask to local position /JA
            swarm_vector = gbest | lbest

            c1 = random.randint(0, max_pos) #random weighting /JA
            c2 = random.randint(0, max_pos) #random weighting /JA
            d1 = swarm[i].position ^ swarm[i].best_position
            d2 = swarm[i].position ^ swarm_vector
            velocity = (d1 & c1) | (d2 & c2)

            # Apply velocity limits
            v = [int(x) for x in bin(velocity)[2:]]
            ones = []
            for n, bit in enumerate(v):
                if bit == 1:
                    ones.append(n)

            while len(ones) > Vmax: #limit number of 1 bits to Vmax /JA
                n = random.randint(0, len(ones) - 1)
                v[ones[n]] = 0
                ones.pop(n)

            v_str = ''
            for bit in v:
                v_str += str(bit)
            v_new = int(v_str, 2)

            if v_new == 0: #if the new velocity is 0, set random bit to 1 /JA
                v_new = 2 ** (random.randint(0, n_vars - 1))

            swarm[i].velocity = v_new

            # Update Position
            swarm[i].position = swarm[i].position ^ swarm[i].velocity

            # Update Cost
            if swarm[i].position in Positions_dic: #if position has already been evaluated, get results /JA
                swarm[i].cost = Positions_dic[swarm[i].position]
            
            # if results exists, read and analyze /JA
            #elif os.path.exists(output_loc+"/Parasitic_S11_" + str(swarm[i].position) + ".csv"):
            #    swarm[i].cost = analyze(swarm[i].position)
            #    Positions_dic[swarm[i].position] = swarm[i].cost
            
            else:
                mat_path = os.path.join(iter_dir,"Particle_{}_matrix.csv".format(swarm[i].position))
                write_matrix_file(update_matrix_from_position(swarm[i].position,matrix0,ind_adj),mat_path)
                swarm[i].cost = evaluate(swarm[i].position, matrix0,ind_adj)
                Positions_dic[swarm[i].position] = swarm[i].cost

            # Update Personal Best
            if swarm[i].cost < swarm[i].best_cost:
                swarm[i].best_position = swarm[i].position
                swarm[i].best_cost = swarm[i].cost
            # Update global best
            if swarm[i].best_cost < global_best_cost:
                global_best_cost = swarm[i].best_cost
                global_best_position = swarm[i].best_position

            # Convergence check
            if i == 0:
                best_costs = swarm[i].best_cost
            if swarm[i].best_cost != best_costs: #All particles must have encountered the best cost for convergence /JA
                convergence_test = 0

        # Update local best
        for i in range(pop):
            for n in [-1,0,1]:
                ind = (i+n) % pop
                if swarm[ind].best_cost < swarm[i].local_best_cost:
                    swarm[i].local_best_cost = swarm[ind].best_cost
                    swarm[i].local_best_position = swarm[ind].best_position

        iter_end = time.time() #iteration end time

        oProject.Save() #Save Project; hypothesis that saving clears RAM and speeds up execution speed /JA

        #Calculate and display iteration duration
        #TO-DO: log iteration duration to swarm data
        iter_duration = iter_end - iter_start #iteration duration in seconds /JA
        msg = "Iteration " + str(j+1) + " duration: " + str(iter_duration) + " seconds"
        hfss_message(project_name, design_name, 0, msg)

        '''
        #Store swarm Data
        with open(output_loc+"/swarm_iteration_" + str(j+1) + ".txt", "w") as swarm_data:
            str_out = optim_param_header(max_it, pop, n_vars, Vmax, temp)
            str_out += "\n\n"
            str_out += swarm_data_string(swarm, j+1, temp, global_best_cost, global_best_position, convergence_test)
            swarm_data.write(str_out)
        '''

        # Exit if converged
        if convergence_test == 1:
            return global_best_cost, global_best_matrix
    return global_best_cost, global_best_matrix

"""
# imported from matrix.py
def write_matrix_file(m,path):
    with open(path,mode='wb') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(m)

def matrix_find(m, criteria):
    '''
    Find elements of 'm' that satisfy 'criteria'
    'm' is a matrix (i.e. a 2D list)
    Returns a list of tuples, each are a row/column index pair
    '''
    ind = [(i,j) for i,r in enumerate(m) for j,c in enumerate(r) if criteria(c)]
    return ind

def matrix_edge_finder(m):
    '''
    finds indices of entries (laterally or diagonally) adjacent to non-zero entries of 'm'.\n
    'm' is a 2D list
    Returns a list of tuples of row/column index pairs
    '''
    ind_non0 = matrix_find(m,lambda x: x>0)
    row_size = len(m)
    col_size = len(m[0])

    ind_adjacent = []
    for x in [-1,0,1]:
        for y in [-1,0,1]:
                for ind in ind_non0:
                    r = ind[0] + x
                    c = ind[1] + y

                    bound1 = r >= 0 and r < row_size
                    bound2 = c >= 0 and c < col_size
                    unique1 = (r,c) not in ind_adjacent
                    unique2 = (r,c) not in ind_non0

                    if bound1 and bound2 and unique1 and unique2:
                        ind_adjacent.append([r,c])

    #convert to list of tuple index pairs; likley unnecessary
    ind_adjacent = [tuple(ind) for ind in ind_adjacent]
    
    return ind_adjacent

def update_matrix_from_position(position, matrix, ind_adj):
    '''
    The entries of 'matrix' at the indices contained in 'ind_adj'
    are replaced with the bits of 'position'.

    'position' and 'ind_adj' are corresponding lists
    '''
    pos_bits = [int(x) for x in bin(position)[2:]] #integer converted into list of integre bits

    for bit,ind in zip(pos_bits,ind_adj):
        matrix[ind[0]][ind[1]] = bit
    
    return matrix
"""
def padMatrix(m, pad_sentinel):
    #do padding
    ind_adj = matrix_edge_finder(m) # get current adjacent indices
    for ind_pair in ind_adj:
        #replace with padding sentinel value
        m[ind_pair[0]][ind_pair[1]] = pad_sentinel

    return m        

def addFromFrame(m,frame, pad_sentinel):
    ind_adj = matrix_edge_finder(m,pad_sentinel) # get current adjacent indices
    frame_ind1 = matrix_find(frame,lambda x: x == 1) # find indices of 1's
    
    # set of indices both currently adjacent and having a 1 in the frame matrix
    frame_and_adj = [ind for ind in frame_ind1 if ind in ind_adj] 

    for ind in frame_and_adj:
        m[ind[0]][ind[1]] = frame[ind[0]][ind[1]]

    return m

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

def evaluate(position, matrix, ind_adj):
    '''
    Configures an HFSS model and simulates, returning the -10dB bandwidth
    '''
    new_matrix = update_matrix_from_position(position,matrix,ind_adj)

    config_ifa(new_matrix)

    sim_start = time.time()
    oDesign.Analyze("Setup1")
    sim_end = time.time()
    sim_duration  = sim_end - sim_start
    hfss_message(project_name, design_name, 0, "Configuration " + str(position) +" simulation duration: " + str(sim_duration))

    oReportModule = oDesign.GetModule("ReportSetup")
    oReportModule.ExportToFile('Output Variables Table 1', temp_csv)
    cost = analyze(temp_csv)
    return cost

def analyze(path):
    # Read simulation results file
    #global output_loc
    
    with open(path) as csvFile:
        reader = csv.reader(csvFile)
        row1 = next(reader) #get header row
        bw_header = row1[1] #get header to extract units
        row2 = next(reader) #get first data row
        bandwidth = row2[1] #get bandwidth
    
    # get bandwidth magnitude
    if bandwidth == '':
        return 0
    else:
        bandwidth = float(bandwidth)

    # extract bandwidth units from header
    units = ['', 'k', 'M', 'G', 'T'] # list of unit prefixes
    order = [1, 3, 6, 9, 12] # associated order of magnitudes
    order_dict = dict(zip(units,order)) # constructing dictionary
    
    match = re.search("\[(.?)Hz\]",bw_header)
    unit = match.group(1)
    bandwidth = bandwidth * 10**order_dict[unit] #convert to Hertz
    return bandwidth / -10**6 #convert to MHz and negate

def optim_param_header(max_it, pop, n_vars, Vmax, lbest, max_gen):
    str_out = "Optimization Parameters"
    str_out += "\n-------------------------"
    str_out += "\nmax_it: " + str(max_it)
    str_out += "\npop: " + str(pop)
    str_out += "\nn_vars: " + str(n_vars)
    str_out += "\nVmax: " + str(Vmax)
    str_out += "\nlbest: " + str(lbest)

    return str_out

def swarm_data_string(swarm, iteration, lbest, global_best_cost, global_best_position, convergence):
    str_out = "Swarm Data"
    str_out += "\nIteration: " + str(iteration)
    str_out += "\n------------"
    for n,p in enumerate(swarm):
        str_out += "\nParticle "+str(n)+":"
        str_out += particle_to_str(p)
        if lbest:
            str_out+= "Local Best Cost: " + str(p.local_best_cost)
        str_out += "\n"
    str_out += "\n\nGlobal Best Cost: " + str(global_best_cost)
    str_out += "\nGlobal Best Position: " + str(global_best_position)
    str_out += "\nConvergence: "
    if convergence:
        str_out += "True"
    else:
        str_out += "False"

    return str_out

def hfss_message(proj, design, priority, msg):
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

if __name__ == "__main__": #allows functions to be defined at end-of-file
    main()
