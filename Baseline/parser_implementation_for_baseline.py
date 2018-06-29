import random
import os

####PARSER_Parameters_(set_here)####
threshold = 1.0                    #
init_weight = 1.0                  #
weight_added = 0.5                 #
decay_param = -0.0001              #
interf_value = -0.00001            #
####################################

class PARSERModel:
    """Python implementation of PARSER (Perruchet & Vinter, 1998)"""
    def __init__(self, thresh, init_w, weight, decay_p, interf):
        self.thresh = thresh
        self.weight = weight
        self.init_w = init_w
        self.decay_p = decay_p
        self.interf = interf
        self.SU = {}
        self.shal_pars = ''

    def process(self, utterance):
        """Processes utterance as a list of shaping units. Each percept created
           within this function corresponds to a single time-step."""

        su_list = self.find_shaping_units(utterance)

        while su_list:
            if len(su_list) == 1:
                percept = su_list[:]
                del su_list[0]
            elif len(su_list) == 2:
                #If there are only two units, percept size cannot be 3.
                #Thus, sizes 1 and 2 have an equal chance of being chosen.
                #(Discussed with Perruchet via e-mail)
                percept_size = random.randint(1,2)
                percept = su_list[0:percept_size]
                del su_list[0:percept_size]
            else:
                percept_size = random.randint(1,3)
                percept = su_list[0:percept_size]
                del su_list[0:percept_size]
                
            self.decay()
            self.interference(percept)
            self.add_shaping_unit(percept)

        self.clean_units()
        
    def find_shaping_units(self,utt):
        """Identifies the shaping units for the utterance (results would be the
           same if the code processed the utterance incrementally -- currently
           implemented as a whole-utterance process to reduce runtime)."""
        su_list = []
        while utt:
            for i in range(len(utt),0,-1):
                if i == 1:
                    su_list.append(utt[0])
                    del utt[0]
                elif ' '.join(utt[0:i]) in self.SU:
                    if self.SU[' '.join(utt[0:i])] >= threshold:
                        su_list.append(' '.join(utt[0:i]))
                        del utt[0:i]
                        break

        #stores the shallow parse of the most recently encountered utterance
        #this is then compared to the output of an actual shallow parser
        self.shal_pars = ' || '.join(su_list)

        return su_list
                        
    def clean_units(self):
        """Removes shaping units with activation rates of 0 or less."""
        removal_list = []
        for item in self.SU:
            if self.SU[item] <= 0:
                removal_list.append(item)
        for item in removal_list:
            del self.SU[item]
    
    def add_shaping_unit(self, percept):
        """Adds a new shaping unit. If the unit already exists, its activation rate
           is simply incremented. Adds weight to its components."""
        per_str = ' '.join(percept)
        if per_str in self.SU:
            self.SU[per_str] += self.weight
        else:
            self.SU[per_str] = self.init_w

        if len(percept) > 1:
            for unit in percept:
                if unit in self.SU:
                    self.SU[unit] += self.weight
                    #Because components are interfered with in a prior processing
                    #stage, we undo interference for percept components here for
                    #the sake of simplicity:
                    self.SU[unit] -= self.interf

    def decay(self):
        """Decay of shaping unit activation rates at the end of each time step.
           Newly discovered units don't decay (P&V, 1998, pg. 252). Thus, this
           action is performed prior to updating of the shaping units."""
        for item in self.SU:
            self.SU[item] += self.decay_p

    def interference(self, percept):
        """Interference with all shaping units containing primitives in the
           current percept. A unit in the percept shaper can be interfered with
           more than once in the same time-step (Perruchet, personal
           communication)."""
        per_lst = ' '.join(percept)
        per_lst = per_lst.split()
        for item in self.SU:
            for unit in item.split():
                if unit in per_lst:
                    self.SU[item] += self.interf

def shallow_parse(filename):
    global threshold
    global init_weight
    global weight_added
    global decay_param
    global interf_value

    model = PARSERModel(threshold, init_weight, weight_added, decay_param, interf_value)

    file = open(filename,'r')
    lines = file.readlines()
    file.close()

    shallow_parse_attempts = []

    for line in lines:
        utterance = line.replace(' || ',' ').split()
        model.process(utterance)
        shallow_parse_attempts.append(model.shal_pars)


rootdir='//home/path/to/corpora'

#Use os.walk to run all corpora in a directory
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if ('.parsed' in file):
            textfile = subdir+'/'+file
            shallow_parse(textfile)
