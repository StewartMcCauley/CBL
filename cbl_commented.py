import random
import os

class MeanDict(dict):
    """Dictionary designed to allow easy calculation
    of the running average of its values."""
    def __init__(self):
        self._total = 0.0
        self._count = 0

    def __setitem__(self, k, v):
        if k in self:
            self._total -= self[k]
            self._count -= 1
        dict.__setitem__(self, k, v)
        self._total += v
        self._count += 1

    def __delitem__(self, k):
        v = self[k]
        dict.__delitem__(self, k)
        self._total -= v
        self._count -= 1

    def average(self):
        if self._count:
            return self._total/self._count

class CBL_Model:
    """Chunk-based Learner (CBL) model. McCauley &
    Christiansen (2011, 2014, submitted)"""
    def __init__(self):
        self.avg_tp = 0
        self.Unigrams = {}
        self.Bigrams = {}
        self.TPRunAvg = MeanDict() 
        self.UniChunks = {}
        self.BiChunks = {}
        self.spa = 0
        self.prod_attempts = 0
        self.ChunkWordPairs = {}
        self.shallow_parses = []

    def add_unigram(self, word):
        """Updates low-level frequency info for unigrams."""
        if word in self.Unigrams:
            self.Unigrams[word] += 1
        else:
            self.Unigrams[word] = 1

    def add_bigram(self, w1, w2):
        """Updates low-level frequency info for bigrams."""
        if w1+' '+w2 in self.Bigrams:
            self.Bigrams[w1+' '+w2] += 1
        else:
            self.Bigrams[w1+' '+w2] = 1

    def add_unichunk(self, chunk):
        """Updates chunkatory for a given chunk."""
        if chunk in self.UniChunks:
            self.UniChunks[chunk] += 1
        else:
            self.UniChunks[chunk] = 1

    def add_bichunk(self, chunk1, chunk2):
        """Updates frequency info for adjacent chunks,
        allowing the model to calculate TP between chunks."""
        if (chunk1, chunk2) in self.BiChunks:
            self.BiChunks[chunk1, chunk2] += 1
        else:
            self.BiChunks[chunk1, chunk2] = 1

    def add_chunkWordPair(self, w1, w2):
        """Updates frequency info for adjacent words occurring as part
        (or all) of a chunk, supporting discovery of new chunks."""
        if (w1, w2) in self.ChunkWordPairs:
            self.ChunkWordPairs[(w1,w2)] += 1
        else:
            self.ChunkWordPairs[(w1,w2)] = 1

    def calc_btp(self, w1, w2):
        """Calculates transition probabilities between words."""
        return float(self.Bigrams[w1+' '+w2])/float(self.Unigrams[w2])

    def update_chunks(self, linelist):
        """Update the chunkatory on-line."""
        line = ' '.join(linelist)
        chunks = line.split(' || ')
        self.add_unichunk(chunks[-1])

        if (len(chunks) > 1):
            self.add_bichunk(chunks[-2],chunks[-1])
    
    def calc_btp_chunks(self, chunk1, chunk2):
        """Calculates the TP between two chunks."""
        if (chunk1, chunk2) in self.BiChunks:
            return float(self.BiChunks[(chunk1, chunk2)])/float(self.UniChunks[chunk2])
        else:
            return 0.0

    def end_of_utterance(self,linelist):
        """End-of-line housekeeping. Adds a chunk-to-chunk
        frequency count for the start-of-utterance marker
        leading into the initial chunk in the utterance.
        This is done last for simplicity -- the result
        would be the same if it were done at the beginning."""
        line = ' '.join(linelist)
        chunks = line.split(' || ')
        self.add_bichunk('#',chunks[0])

    def bag_o_chunks(self, line):
        """Yields a bag-of-chunks for the production task."""
        bag = []
        while line:
            for i in range(len(line),0,-1):
                if i == 1:
                    bag.append(line[0])
                    del line[0]
                    break
                elif ' '.join(line[0:i]) in self.UniChunks:
                    bag.append(' '.join(line[0:i]))
                    del line[0:i]
                    break
        return bag

    def upd_run_avg(self, tp, prev_word, item):
        """Update the running average TP."""
        if self.Unigrams[item] > 1:
            if prev_word != '#':
                self.TPRunAvg[prev_word, item] = tp
                self.avg_tp = self.TPRunAvg.average()

    def parse(self, tp, shal_pars, prev_word, item):
            """Shallow parsing operations for this timestep."""
            if tp < self.avg_tp:
                if (prev_word, item) not in self.ChunkWordPairs:
                    self.update_chunks(shal_pars)
                    if prev_word != '#':
                        shal_pars.append('||')
                elif self.ChunkWordPairs[prev_word, item] < 2:
                    self.update_chunks(shal_pars)
                    if prev_word != '#':
                        shal_pars.append('||')
            else:
                self.add_chunkWordPair(prev_word,item)

            return shal_pars
    
    def big_spa(self, utterance):
        """Implements our modified version of the bag-of-words
        incremental sentence generation task of Chang et al. 2008"""
        self.prod_attempts += 1
        line = utterance.split()
        del line[0] #delete speaker tag (*CHI:)
        del line[-1] #delete punctuation
        bag = self.bag_o_chunks(line[:])

        prev_chunk = '#' #set start of utterance marker as first chunk

        produced = []

        while bag: #incrementally produce new utterance chunk-by-chunk
            highest = 0.0
            candidates = []
            for item in bag:
                tp = self.calc_btp_chunks(prev_chunk, item)
                if tp > highest:
                    candidates = [item]
                    highest = tp
                elif tp == highest:
                    candidates.append(item)
            output = candidates[random.randint(0,len(candidates)-1)]

            produced.append(output)
            bag.remove(output)
            prev_chunk = output

        if ' '.join(produced) == ' '.join(line):
            self.spa += 1
            
    def process(self, utterance):
        """Process an utterance on-line, treating each
        word as a separate timestep."""
        line = utterance.split()
        #if list has less than 3 elements, it is empty b/c
        #auto-cleaning removed a non-speech sound, etc.
        if len(line) < 3:
            return

        #if it's a child utterance, run it through BIG-SPA
        if line[0] == '*CHI:':
            if len(line) > 3:
                self.big_spa(utterance)

        del line[0] #remove speaker tag
        del line[-1] #remove punctuation
        prev_chunk = '#'#set start of utterance marker
        prev_word = '#'#set start of utterance marker
        self.add_unigram(prev_word)
        self.add_unichunk(prev_chunk)

        shal_pars = []
        
        for item in line:
            """Process incrementally, on-line --
            each word represents a time-step."""
            #update low-level freq info
            self.add_unigram(item)
            self.add_bigram(prev_word, item)

            #calculate running average TP
            tp = self.calc_btp(prev_word, item)
            self.upd_run_avg(tp, prev_word, item)

            #shallow parse the utterance
            shal_pars = self.parse(tp, shal_pars, prev_word, item)
            
            shal_pars.append(item)
            prev_word = item
        
        #end of utterance has been encountered, so the final
        #chunk can now be updated online
        self.update_chunks(shal_pars)
        
        #house-keeping operation
        self.end_of_utterance(shal_pars)

        #Add the on-line shallow parse to a list --     
        #this list can then be compared to the gold standard
        #shallow parses.
        self.shallow_parses.append(shal_pars)    


rootdir='//home/directory/to/corpora'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if ('.capp' in file):
            textfile = subdir+'/'+file
            file = open(textfile,'r')
            lines = file.readlines()
            file.close()

            production_scores = []
            
            for i in range(0,10):#define number of iterations (for averaging)
                
                model = CBL_Model()

                for line in lines:
                    #Ensure age-markers aren't treated as utterances
                    #Ex. "*AGEIS: 25 .\n" <-- child at 25 months
                    if '*AGEIS:' in line:
                        continue
                    #Process the utterance incrementally and on-line
                    model.process(line)

                production_scores.append(float(model.spa)/float(model.prod_attempts))
                    
            #Print out the model's average score on the production task
            print textfile, str(sum(production_scores) / float(len(production_scores)))

            #output record of the on-line shallow parses to a new file for scoring
            #against gold-standard shallow parses
            outputfile = open(textfile.split('.capp')[0]+'.sh.parses','w')
            for item in model.shallow_parses:
                outputfile.write(' '.join(item)+'\n')
