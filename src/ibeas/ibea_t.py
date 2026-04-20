
import numpy as np

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
#from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population

from ibea_callbacks import MyCallback
from ibea import IBEA

class IBEATest:
#class IBEAAlgos: #New name
    '''
    This class centralize the calling of my different algorithms.
    '''

    @staticmethod
    def __ibea(Advance, problem, K, pop_size, n_gen, **kwargs):
        #TODO: add selection to all Advance
        if hasattr(Advance, 'selection'):
            selection = Advance.selection() 
        else:
            selection = TournamentSelection(func_comp=ibea_binary_tournament)

        algorithm = IBEA(Advance,
                         pop_size=pop_size,
                         K = K,
                         sampling=FloatRandomSampling(),
                         selection=selection,
                         crossover=SBX(eta=20, prob=0.9),
                         mutation=PM(prob=1/problem.n_var, eta=20),
                         output=MultiObjectiveOutput(),
                         **kwargs)
        criterion = 'n_gen'
        if 'stop_criterion' in kwargs:
            criterion = kwargs.get('stop_criterion')

        termination = (criterion, n_gen)
        algorithm.termination = termination
        #algorithm.setup(problem=problem, callback=MyCallback(), **kwargs)
        algorithm.setup(problem=problem, **kwargs)
        #Une technique pour définir un _infill personnalisé
        if hasattr(Advance, "_infill"):
            Advance._infill(algorithm, **kwargs)
        res = algorithm.run()
        return res, algorithm, n_gen

    @staticmethod
    def ibea_original(problem=None, K=0.1, pop_size=100, n_gen=8):
        #TODO: to change for, "from ibea.advance import Advance"
        from ibeas.ibea1.ibea_advance import Advance                 #Original algorithm (zitzler2004) (ibea1)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen)

    @staticmethod
    def ibea_adaptive_original(problem=None, K=0.1, pop_size=100, n_gen=8):
        from ibeas.ibea3.ibea_advance_3 import Advance               #Adaptive IBEA (zitzler2004) (ibea3)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen)

    @staticmethod
    def ibea_1a(problem=None, K=0.1, pop_size=100, n_gen=8):
        from ibeas.ibea1a.ibea_advance_1a import Advance             #  (ibea1a)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen)

    @staticmethod
    def ibea_1b(problem=None, K=0.1, pop_size=100, n_gen=8):
        from ibeas.ibea1b.ibea_advance_1b import Advance             #  (ibea1b)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen)

    @staticmethod
    def ibea_2(problem=None, K=0.1, pop_size=100, n_gen=8):
        from ibeas.ibea2.ibea_advance_2 import Advance               #  (ibea2)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen)

    @staticmethod
    def ibea_4(problem=None, K=0.1, pop_size=100, n_gen=8, **kwargs):
        from ibeas.nibea.ibea_advance_4 import Advance               #  (nibea)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen, **kwargs)

    @staticmethod
    def ibea_4a(problem=None, K=0.1, pop_size=100, n_gen=8, **kwargs):
        from ibeas.nibea.ibea_advance_4a import Advance              #  (nibea_a)
        return IBEATest.__ibea(Advance, problem, K, pop_size, n_gen, **kwargs)




    # For testing
    @staticmethod
    def main(problem=None, K=1, pop_size=100, n_gen=8):
        # Demo to show and analyse what happens with this calling...
        from ibeas.ibea1.ibea_advance import Advance
        algorithm = IBEA(Advance,
                         pop_size=pop_size,
                         K = K,
                         save_history=True,
                         #archive=list(),
                         seed=1,
                         #sampling=FloatRandomSampling(),
                         selection=TournamentSelection(func_comp=ibea_binary_tournament),
                         sampling=np.array(SAMPLING1_PROB1),
                         #selection=RandomSelection(), #For testing purpose
                         #selection=TournamentSelection(func_comp=binary_tournament), # From NSGA2
                         #selection = TournamentSelection(pressure=2, func_comp=binary_tournament), #voir: https://pymoo.org/operators/selection.html
                         crossover=SBX(eta=15, prob=0.9),
                         mutation=PM(eta=20),
                         #eliminate_duplicates = NoDuplicateElimination(),
                         output=MultiObjectiveOutput()) # TODO: Incomplete calling
        #algorithm.termination = DefaultMultiObjectiveTermination()
        algorithm.termination = ('n_gen', n_gen)
        #algorithm.tournament_type = 'comp_by_dom_and_crowding'
        # With just this call, nothing happens except the init of the algo...
        algorithm.setup(problem=problem, callback=MyCallback())
        res = algorithm.run()
        return res, algorithm, n_gen

