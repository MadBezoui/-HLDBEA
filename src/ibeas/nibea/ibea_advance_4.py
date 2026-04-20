import types

import numpy as np

from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

from ibeas.nibea.custominfill import CustomInfill
from ibeas.nibea.temppopmethod import TempPopMethod
from ibeas.nibea.selectpopmethod import SelectPopMethod
from ibeas.nibea.envselectionmethod import EnvSelectionMethod
from ibeas.nibea.scores import CalcScore
from ibeas.nibea.util import Advance4Util
#from ibeas.nibea.nibea_util import EnvSelectionMethod

#from ibea_util import IBEAUtilsOriginalHV as ibu
from util.misc import get_caller_function


# ---------------------------------------------------------------------------
# Stagnation detection (paper Section 3.4)
# ---------------------------------------------------------------------------
# Parameters (from the paper):
#   theta_restart = 0.95   : fraction of zero-score individuals that triggers the check
#   G_stag        = 20     : number of consecutive generations to monitor HV
#   delta_HV      = 1e-4   : minimum HV improvement to NOT consider stagnation
#   rho           = 0.20   : fraction of the population to restart randomly
# ---------------------------------------------------------------------------
_THETA_RESTART = 0.95
_G_STAG        = 20
_DELTA_HV      = 1e-4
_RHO           = 0.20


def _check_and_apply_restart(algorithm):
    """
    Implements the stagnation detection and restart mechanism (paper Section 3.4).

    Triggered when:
        1. >= theta_restart * N individuals have fitness == 0  (locally non-dominated)
        2. The HV improvement over the last G_stag generations is < delta_HV

    Action: replace rho * N individuals by uniform random samples in [xl, xu].
    """
    Ff = algorithm.pop.get("Fit")
    n_zero = int(np.sum(Ff >= 0))   # fit=0 are the best (locally non-dominated)
    frac_zero = n_zero / len(algorithm.pop)

    # Condition 1: enough zero-score individuals
    if frac_zero < _THETA_RESTART:
        return

    # Ensure history list exists
    if not hasattr(algorithm, '_hv_history'):
        algorithm._hv_history = []

    # Compute current HV
    try:
        from pymoo.indicators.hv import HV
        ref_point = algorithm.problem.nadir_point()
        ind = HV(ref_point=ref_point)
        hv_current = float(ind(algorithm.pop.get('F')))
    except Exception:
        # If HV cannot be computed (e.g., no nadir), skip restart
        return

    algorithm._hv_history.append(hv_current)

    # Keep only the last G_stag values
    if len(algorithm._hv_history) > _G_STAG:
        algorithm._hv_history.pop(0)

    # Condition 2: check stagnation only when we have enough history
    if len(algorithm._hv_history) < _G_STAG:
        return

    delta = abs(algorithm._hv_history[-1] - algorithm._hv_history[0])
    if delta >= _DELTA_HV:
        return

    # --- Stagnation confirmed: restart rho * N individuals ---
    n_restart = max(1, int(_RHO * algorithm.init_pop_size))
    problem = algorithm.problem

    # Sample new random points
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    sampler = FloatRandomSampling()
    new_pop = sampler(problem, n_restart)
    algorithm.evaluator.eval(problem, new_pop, algorithm=algorithm)
    new_pop.set("Fit",    np.zeros(n_restart))
    new_pop.set("FitMin", np.zeros(n_restart))
    new_pop.set("FitLife", np.zeros(n_restart))

    # Replace the worst rho*N individuals (lowest Fit values)
    Ff_sorted = np.argsort(Ff)                # ascending: worst first
    keep_idx = Ff_sorted[n_restart:]          # keep the best (N - n_restart)
    survived  = algorithm.pop[keep_idx]
    algorithm.pop = Population.merge(survived, new_pop)

    # Reset HV history after restart
    algorithm._hv_history = []
    print(f"[HLDBEA] Restart triggered at gen {algorithm.n_gen}: "
          f"{n_restart} individuals replaced (delta_HV={delta:.2e}).")


class Advance4Base:
    """
    Base class with core logic for population fitness updates and selection initialization.
    Provides common functionality for fitness assignment, population merging, 
    and selection method dispatch in MOEAs.
    """

    @staticmethod
    def selection(**kwargs):
        from ibea_selection import nibea_binary_tournament
        #from ibea_selection import dom_first_binary_tournament
        print('...Using custom selection')
        return TournamentSelection(func_comp=nibea_binary_tournament)
        #return TournamentSelection(func_comp=dom_first_binary_tournament)

    @classmethod
    def _initialize_advance(cls, algorithm, infills=None, **kwargs):
        """
        Sets paper-compliant defaults for all HLDBEA hyper-parameters and performs
        the initial fitness assignment.

        Paper defaults (ACT_Ghani_V2-6, Section 3):
            sc_method      = "0"      → fit(A) = -score(A) - λ·dist(A)
            env_sel_method = "crowd"  → Rank & Crowding as tie-breaker (Algorithm 4)
            sel_pop        = "2"      → iterative fitness recalculation
            temp_pop       = "0"      → standard merge P ∪ P'
            exact_method   = "epsilon"→ ε-constraint for exact infill (Algorithm 3)
            tau            = 1        → size of elite set for exact infill (Algorithm 3)
            lmbda          = 0.1      → distance weight in fitness (already set in IBEA.__init__)
        """
        # sc_method: "0" is paper-compliant (augmented fitness)
        if not hasattr(algorithm, "fitm"):
            print('Attention: Default FitMin to True')
            algorithm.fitm = True
        if not hasattr(algorithm, "sc_method"):
            print('Attention: Default Score method → "0" (paper-compliant)')
            algorithm.sc_method = "0"
        if not hasattr(algorithm, "env_sel_method"):
            print('Attention: Default ENV Selection → "crowd" (paper: Rank & Crowding)')
            algorithm.env_sel_method = "crowd"
        if not hasattr(algorithm, "sel_pop"):
            print('Attention: Default Sel pop → "2"')
            algorithm.sel_pop = "2"
        if not hasattr(algorithm, "temp_pop"):
            print('Attention: Default Temp pop → "0"')
            algorithm.temp_pop = "0"
        if not hasattr(algorithm, "exact_method"):
            print('Attention: Default Exact method → "epsilon"')
            algorithm.exact_method = "epsilon"
        # tau: number of elite individuals to apply exact infill (Algorithm 3 of the paper)
        if not hasattr(algorithm, "tau"):
            algorithm.tau = 1
        # lmbda: may have been set in IBEA.__init__; ensure it exists here too
        if not hasattr(algorithm, "lmbda"):
            algorithm.lmbda = 0.1

        print(f"...Using adaptive_k: {hasattr(algorithm, 'adaptive_k')}")
        print(f"...Using FitMin: {algorithm.fitm}")
        print(f"...Using TempPopMethod: {algorithm.temp_pop}")
        print(f"...Using SelectPop: {algorithm.sel_pop}")
        print(f"...Using EnvSelectionMethod: {algorithm.env_sel_method}")
        print(f"...Using sc_method: {algorithm.sc_method}")
        print(f"...Using exact_method: {algorithm.exact_method}")
        print(f"...Using tau (elite set size): {algorithm.tau}")
        print(f"...Using lambda (distance weight): {algorithm.lmbda}")

        # Compute and set Fitness
        _F = algorithm.pop.get("F")
        Ff = Advance4Util.calc_scores(algorithm, _F)
        algorithm.pop.set("Fit", Ff)

        #algorithm.pop.set("FitCumul", Ff)
        if algorithm.fitm:
            algorithm.pop.set("FitMin", Ff)
        #TODO: not used yet 11/06/2025
        algorithm.pop.set("FitLife", np.zeros(len(algorithm.pop)))

        # Compter le nombre de fois d'appel de sel_pop (et déduire par la suite env_sel)
        algorithm.count_sel_pop = 0

    @staticmethod
    def _temp_pop(algorithm, infills, **kwargs):
        if algorithm.temp_pop == '2':
            TempPopMethod._temp_pop_2(algorithm, infills, **kwargs)
        elif algorithm.temp_pop == '1':
            TempPopMethod._temp_pop_1(algorithm, infills, **kwargs)
        else : # '0', ...
            TempPopMethod._temp_pop_0(algorithm, infills, **kwargs)

    @classmethod #Not used yet. Ne doit pas avoir pasx et pasy comme params...
    def calc_fitness(cls, algorithm, _F):
        return Advance4Util.calc_scores(algorithm, _F)

    @staticmethod
    def do_sel_pop(algorithm, _F, FfSorted, fn1):
        sel_pop = algorithm.sel_pop
        if  (sel_pop == "0") or (fn1 == algorithm.init_pop_size):
            SelectPopMethod.sel_pop0(algorithm, _F, FfSorted)
        elif sel_pop == "1":
            SelectPopMethod.sel_pop1(algorithm, FfSorted, fn1)
        elif sel_pop == "3": #Remove worst
            SelectPopMethod.sel_pop3(algorithm, FfSorted, fn1)
        elif sel_pop == "4": #select best and continue with ranking
            SelectPopMethod.sel_pop4(algorithm, FfSorted, fn1)
        else: #sel_pop == "2": #select best
            SelectPopMethod.sel_pop2(algorithm, FfSorted, fn1)

    @staticmethod
    def do_env_selection(algorithm, _F, FfSorted, fn1):
        """
        Dispatches to the appropriate environmental selection method.

        Paper (Algorithm 4): when more than N individuals have fitness=0,
        use NSGA-II Rank & Crowding ("crowd") as the tie-breaker.
        """
        env_sel_method = algorithm.env_sel_method
        if env_sel_method == "0":
            EnvSelectionMethod.method0(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "1":
            EnvSelectionMethod.method1(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "2":
            EnvSelectionMethod.method2(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "rand":
            EnvSelectionMethod.method_rand(algorithm, _F, FfSorted, fn1) # -
        elif env_sel_method == "crowd":
            # Paper-compliant: Rank & Crowding tie-breaker (Algorithm 4)
            EnvSelectionMethod.method_crowd(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "exact":
            EnvSelectionExactMethod.method_exact(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "rank":
            EnvSelectionMethod.method_rank(algorithm, _F, FfSorted, fn1)
        elif env_sel_method == "inv":
            EnvSelectionMethod.method_inv(algorithm, _F, FfSorted, fn1)
        else: # default: paper-compliant crowd
            EnvSelectionMethod.method_crowd(algorithm, _F, FfSorted, fn1)


class Advance4Common(Advance4Base):

    @classmethod
    def update_population_fitness(cls, algorithm):
        Ff = Advance4Util.calc_scores(algorithm)
        #algorithm.pop.set("Fit", Ff)
        #
        #TODO: idea "id-1"
        #algorithm.pop.set("FitCumul", ...)
        #algorithm.pop.set("FitLife", ...)
        if algorithm.fitm:
            fmin = algorithm.pop.get("FitMin") # TODO: exist pr t lé pts ?
            algorithm.pop.set("FitMin", np.minimum(fmin, Ff))
            algorithm.pop.set("Fit", np.minimum(fmin, Ff))
        else:
            algorithm.pop.set("Fit", Ff)

        fl = algorithm.pop.get("FitLife") + 1
        algorithm.pop.set("FitLife", fl)
        #print(f"fl: ({algorithm.n_iter})", np.sort(fl))

        #LOG
        #fit_arr = algorithm.pop.get("Fit")
        #print("Fitness: ", fit_arr)
        #unique_elements, counts = np.unique(fit_arr, return_counts=True)
        #print("Unique elements:", unique_elements)
        #print("Counts:", counts)


    @classmethod
    def _advance(cls, algorithm, infills=None, **kwargs):
        '''
        Main iteration step of HLDBEA (Algorithm 2 of the paper).

        Steps:
          1. Merge offspring (infills) with current population P → P̃ = P ∪ P'
          2. Compute fitness for P̃: fit(A) = -score(A) - λ·dist(A)
          3. Environmental selection: keep N best individuals
             a. If |{A ∈ P̃ : fit(A)=0}| < N  → call sel_pop (fills up from next-best)
             b. If |{A ∈ P̃ : fit(A)=0}| > N  → call env_selection (trim with Rank & Crowding)
          4. [Optional] Stagnation detection + restart (paper Section 3.4)

        algorithm.pop : current population P (before the merge)
        infills       : offspring P'
        '''

        # the current population; backup copy.
        pop = algorithm.pop
        # Bug-fix (line 181 original): was 'fills' (NameError), must be 'infills'
        if algorithm.fitm:
            infills.set("FitMin", np.zeros(len(infills)))
        infills.set("FitLife", np.zeros(len(infills)))

        # Add infills to history
        algorithm.Finfills_archive = infills.get("F")

        # Intermediate pop: merge (P + infills) — Step 1
        cls._temp_pop(algorithm, infills, tech='2')
        #Now algorithm.pop = merge (P + infills)

        _F = algorithm.pop.get("F") # After the merge (see pop for the original) 
        # Step 2: Calculate fit(A) = -score(A) - λ·dist(A) for the merged population
        Ff = Advance4Util.calc_scores(algorithm, _F)

        # idea "id-1": keep track of historical minimum fitness
        if algorithm.fitm:
            Ff = np.minimum(Ff, algorithm.pop.get("FitMin"))

        if algorithm.fitm:
            algorithm.pop.set("FitMin", Ff)
        algorithm.pop.set("Fit", Ff)



        ###### SURVIVAL / ENVIRONMENTAL SELECTION (Algorithm 4 of the paper)
        # Goal: select N individuals from P̃ = P ∪ P' (size 2N).
        #   • All individuals with fit=0 (locally non-dominated in P̃) are preferred.
        #   • If their count < N: complete with the next-best (sel_pop).
        #   • If their count > N: trim using Rank & Crowding (env_sel, paper Section 3.3).

        itt = 1000
        if False:
        #if (algorithm.n_iter % itt == 0):
            #Voir NSGA2
            print("Attention: utilisation d'un technique itt")
            from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
            survival = RankAndCrowding()
            pp = algorithm.pop
            pp.set("rk", np.arange(len(pp)))
            rk = pp.get("rk")
            algorithm.pop = survival.do(algorithm.problem, pp,
                              n_survive=algorithm.init_pop_size,
                              algorithm=algorithm)
            rk1 = algorithm.pop.get("rk")
            #print(rk, rk1, np.setdiff1d(rk, rk1))
            #print(algorithm.pop.get("rank"))
            algorithm.eliminated_tab = pp.get("F")[np.setdiff1d(rk, rk1)]
        else:
            FfSorted, fn1 = Advance4Util.get_sorted(Ff) # Les bons points se trouvent à la fin.
            # Count of individuals with the maximal (best) fitness score
            if (len(algorithm.pop) - fn1) < algorithm.init_pop_size:
                algorithm.count_sel_pop = algorithm.count_sel_pop + 1
                cls.do_sel_pop(algorithm, _F, FfSorted, fn1)
            else:
                # More than N individuals have the best fitness → env selection trims
                cls.do_env_selection(algorithm, _F, FfSorted, fn1)

        # Step 4: Stagnation detection + restart (paper Section 3.4)
        _check_and_apply_restart(algorithm)

        #TODO Inutile peut être (?). Les individus ont déjà 'fit' calculée à partir de la pop augmentée:
        # doit-on recalculer avec la new pop ou laisser les anciens?
        #Pour l'instant, ne pas recalculer
        if False:
            #fit est utilisée dans la génération d'un offspring, la recalculer ou pas l'influence donc!
            cls.update_population_fitness(algorithm)

        ##idea: update K with n_iter
        if hasattr(algorithm, "adaptive_k"):
        #if algorithm._adaptive_k:
            #algorithm.K = algorithm.K + algorithm.termination.perc
            algorithm.K = algorithm.K + (algorithm.count_sel_pop - algorithm.n_iter/2)/(algorithm.count_sel_pop+1)

        # LOG: some stats for debugging
        if False:
            #LOG: HV
            from pymoo.indicators.hv import HV
            indh = HV(ref_point=algorithm.problem.nadir_point())
            hv = indh(algorithm.pop.get('F'))
            #print(f"NIBEA: gen = {algorithm.n_gen} Hv = {hv:.{5}f}")

            from pymoo.indicators.igd import IGD
            pf = algorithm.problem.pareto_front()
            indigd = IGD(pf)
            igd = indigd(algorithm.pop.get('F'))
            #print(f"NIBEA: gen = {algorithm.n_gen} IGD = {igd:.{5}f}")

            print(f"NIBEA: gen = {algorithm.n_gen} Hv = {hv:.{5}f} IGD = {igd:.{5}f}")


#TODO: Intégrer NIBEA+ ici.
class AdvanceCustomInfill(Advance4Common):
    """
    Extends HLDBEA with a custom offspring generation strategy.

    The paper (Algorithm 2 + Algorithm 3) describes two tracks:
      • Elite track  : top-τ individuals undergo ε-constraint local improvement (Algorithm 3)
      • Genetic track: remaining individuals undergo SBX + polynomial mutation

    This is activated by setting inf='1a' (τ individuals from custom_infill_1a).
    """

    @staticmethod
    def _infill(algorithm, **kwargs):

        #return # No custom infill
        if not hasattr(algorithm, "inf"):
            algorithm.infill_name = ""
            return

        inf = getattr(algorithm, "inf")
        if inf is None:
            algorithm.infill_name = ""
            return

        print(f"...Using custom infills {inf}")

        #Contient les points générés par la méthode exacte si utilisée. A implémenter ou pas par chaqu méthode...
        algorithm.exact_generated_tab = []
        #Contient les points sélectionnés pour la méthode exacte (ensemble E)
        algorithm.exact_selected_tab = []

        if inf == '0':
            #Définit un autre _infill: mating using just the individuals with fitness=0
            #A utiliser avec _temp_pop = '0'
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_0, algorithm)
            algorithm.infill_name = "0"
        elif inf == '0a':
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_0a, algorithm)
        elif inf == '1':
            #Use exact method
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_1, algorithm)
        elif inf == '1a':
            # PAPER-COMPLIANT: elite τ individuals → ε-constraint (Algorithm 3)
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_paper, algorithm)
            algorithm.infill_name = "1a"
        elif inf == '1a_legacy':
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_1a, algorithm)
            algorithm.infill_name = "1a_legacy"
        elif inf == '1aa':
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_1aa, algorithm)
            algorithm.infill_name = "1aa"
        elif inf == '1b':
            algorithm._infill = types.MethodType(CustomInfill._custom_infill_1b, algorithm)
        else:
            print("WARNING: inf value not valid. No custom infill will be used.")



class AdvanceTT(AdvanceCustomInfill): #TODO
    @staticmethod
    def do_sel_pop(algorithm, _F, pop, psize, FfSorted, fn1): #TODO. See sel_pop_2
        '''
        This method selects the individuals with fitness score equal 0 (?) and then recalculates
        the scores of all the remaining individuals and then doing the selection again.
        '''
        off1  = pop[FfSorted[fn1:]]
        reste = pop[FfSorted[:fn1]]

        while len(off1) < psize:
            _Fr = reste.get("F")
            #print("Old FitMin reste: ", reste.get("FitMin"))
            Ffr = Advance4Util.calc_scores(algorithm, _Fr, reste)
            if algorithm.fitm:
                Ffr = np.minimum(reste.get("FitMin"), Ffr)
                reste.set("FitMin", Ffr)
            reste.set("Fit", Ffr)


            FfSorted, fn1 = Advance4Util.get_sorted(Ffr)
            #print("FfSorted reste", FfSorted)
            #print(fn1)
            m = np.maximum(len(reste) - (psize - len(off1)), fn1)
            #print(m)
            #off1 = Population.merge(off1, reste[FfSorted[m:]])
            off1 = Population.merge(reste[FfSorted[m:]], off1)
            reste = reste[FfSorted[:m]]
            #print(len(reste))
            if len(off1) >= psize or len(reste)==0:
                #off1 = off1[:psize]
                break
        if len(off1) > psize:
            off1 = off1[:psize]
        return off1
        #algorithm.eliminated_tab = reste.get("F")

    @staticmethod
    def do_env_selection__(algorithm, _F, FfSorted, fn1):
        pass

    @classmethod
    def _advance(cls, algorithm, infills=None, **kwargs):
        '''
        . L'idée est de prendre la pop actuelle avant le merge puis les infills
        . et tirer une partie de chacun avec l'env selection.
        . algorithm.pop : current population
        . infills       : offspring
        '''

        # the current population; backup copy.
        pop = algorithm.pop
        infills.set("FitMin", np.zeros(len(infills)))

        # Add infills to history
        algorithm.Finfills_archive = infills.get("F")

        pop_merged = Population.merge(algorithm.pop, infills)

        _F = pop_merged.get("F")
        # Calculate the fitness of pop_merged
        Ff = Advance4Util.calc_scores(algorithm, _F)
        if algorithm.fitm:
            Ff = np.minimum(Ff, pop_merged.get("FitMin"))
        #print("Ff FitMin after merge: ", Ff)

        #Inutile pour sel_pop0 mais à faire pour sel_pop2 !
        pop_merged.set("FitMin", Ff)
        pop_merged.set("Fit", Ff)

        pop_initial = pop_merged[:len(pop)]
        pop_infill  = pop_merged[len(pop):]

        ###### SURVIVAL (Environment selection)
        #TODO: L'idée est de prendre la pop actuelle avant le merge puis les infills
        # et tirer une partie de chacun avec l'env selection.
        Ff_1 = pop_initial.get("Fit")
        Ff_2 = pop_infill.get("Fit")
        #a = len(pop_initial)//2
        a = algorithm.init_pop_size // 2
        b = algorithm.init_pop_size - a
        if len(pop_infill) < b:
            a = a + (b - len(pop_infill))
            b = len(pop_infill)

        FfSorted, fn1 = Advance4Util.get_sorted(Ff_1)
        pop_initial = cls.do_sel_pop(algorithm, _F[len(pop_initial)], pop_initial, a, FfSorted, fn1) #TODO

        FfSorted, fn1 = Advance4Util.get_sorted(Ff_2)
        pop_infill = cls.do_sel_pop(algorithm, _F[len(pop_infill)], pop_infill, b, FfSorted, fn1)
        #pop_infill = pop_infill[FfSorted[(len(pop_infill)//2):]]
        #algorithm.eliminated_tab = _F[FfSorted[:algorithm.init_pop_size]]

        algorithm.pop = Population.merge(pop_initial, pop_infill)
        assert len(algorithm.pop) == algorithm.init_pop_size, f"error:{len(algorithm.pop)}"

        cls.update_population_fitness(algorithm)



#class Advance(Advance4Common): #Méthode de base
class Advance(AdvanceCustomInfill): #Définit un autre _infill
#class Advance(AdvanceTT):
    @classmethod
    def adv_name(cls):
        chain = ''
        for base in cls.__mro__:
            chain += f"{base.__name__}"
        #return chain
        return cls.__mro__[0].__name__ + cls.__mro__[1].__name__
