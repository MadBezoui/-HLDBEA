import numpy as np

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput

from pymoo.operators.selection.tournament import TournamentSelection # Voir: https://pymoo.org/operators/selection.html

from ibea_selection import ibea_binary_tournament


####################################
#### Algorithm
# Générer une population initiale
# Iterate
#   Fitness assignment
#   Environmental selection:
#       Iterate...
#   Mating selection: temporary mating pool P'.
#   Variation: recombination and mutation to P'. Merge the offspring with P.
#
#### Fin algorithm.

class IBEA(GeneticAlgorithm):

    def __init__(self,
                 ibea_advance,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=ibea_binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 output=MultiObjectiveOutput(),
                 K=1,
                 lmbda=0.1,
                 **kwargs):
        """
        Parameters
        ----------
        K     : float
            Neighbourhood radius multiplier.  Paper formula: pₐ = K·(max fₐ - min fₐ)/N.
            The paper fixes K=1 (Section 3.1); kept as a parameter for ablation.
        lmbda : float
            Weight of the distance term in the fitness function (λ in the paper, Section 3.2).
            fit(A) = -score(A) - λ · dist(A).  Default 0.1 as in the paper.
        """
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            output=output,
            **kwargs)

        self.ibea_advance = ibea_advance
        self.init_pop_size = pop_size
        self.K = K
        # λ: weight of the normalised distance-to-ideal term in the fitness (paper Section 3.2)
        self.lmbda = lmbda
        #Save eliminated points in each iteration
        self.eliminated_tab = None
        #Save the infills to be merged with the current pop
        self.Finfills_archive = None
        #Save the merged pop + infills
        self.Fmerged_pop = None

    def _initialize_advance(self, infills=None, **kwargs):
        if False:
            print("********* _initialize_advance (Initial fitness assignment)\nlen(infills)=", len(infills))
        self.ibea_advance._initialize_advance(self, infills=infills, **kwargs)
        if False:
            print("<<<<<<<<< End _initialize_advance\n")


    def _advance(self, infills=None, **kwargs):
        """
          In _advance (advance) we are changing algorithm.pop
        """
        self.ibea_advance._advance(self, infills=infills, **kwargs)
        #LOG: print("<<<<<<<<< End _advance")


