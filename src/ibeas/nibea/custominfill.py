import numpy as np

from pymoo.core.population import Population

from ibeas.nibea.util import Advance4Util

class CustomInfill:
    """
    Collection of custom infill strategies for generating offspring in MOEAs.

    Paper-compliant method (Algorithm 2 + Algorithm 3 of ACT_Ghani_V2-6):
        _custom_infill_paper :
            • Elite set E  = top-τ individuals by fitness (τ = algorithm.tau, default 1)
            • E undergoes ε-constraint local improvement  (Algorithm 3, k_max=2 SLSQP iters)
            • Remaining (P \\ E) generates offspring via SBX + polynomial mutation
            • Offspring = merge(E_improved, genetic_offspring)

    Other methods are retained as experimental variants.
    Les infills sont liés à l'algorithme et à la population globale...
    """

    # ------------------------------------------------------------------
    # PAPER-COMPLIANT METHOD (Algorithm 2 + Algorithm 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _custom_infill_paper(self):
        """
        Paper-compliant offspring generation (Algorithm 2, Section 3.3 of ACT_Ghani_V2-6).

        Two-track generation:
          1. Elite track (Algorithm 3):
               E = top-τ individuals (highest fitness, i.e. fitness closest to 0).
               Each individual in E is improved by the ε-constraint NLP solver
               (SLSQP, k_max=2 iterations, random objective index).
          2. Genetic track:
               The remaining N-τ individuals produce offspring via standard
               SBX crossover + polynomial mutation.

        τ is read from algorithm.tau (default 1, set in _initialize_advance).
        k_max (NLP iterations) is fixed to 2 as specified in the paper.
        """
        assert len(self.pop) == self.init_pop_size, "len(self.pop) != self.init_pop_size"

        Ff = self.pop.get("Fit")
        FfSorted, fn1 = Advance4Util.get_sorted(Ff)

        # τ: size of the elite set (paper Algorithm 3, Section 3.3)
        tau = getattr(self, 'tau', 1)
        # k_max: NLP solver iterations (paper Algorithm 3)
        k_max = 2

        # --- Elite set E = top-τ individuals (highest fitness = least negative) ---
        elite_idx = FfSorted[len(FfSorted) - tau:]   # last τ in sorted order = best
        rest_idx  = FfSorted[:len(FfSorted) - tau]

        elite_pop = self.pop[elite_idx]
        rest_pop  = self.pop[rest_idx]

        # --- Track 1: ε-constraint improvement of each elite individual ---
        from ibeas.nibea.scalarization_solvers import EpsilonConstraintNLP
        pb = self.problem
        list_x_elite = []
        for px in elite_pop.get("X"):
            # Randomly pick the objective to minimise (Algorithm 3, step 3)
            main_idx = self.random_state.integers(0, pb.n_obj)
            r = EpsilonConstraintNLP.solve(pb, px, k_max,
                                           main_obj_index=main_idx,
                                           method='SLSQP')
            list_x_elite.append(r)

        off_elite = Population.new("X", np.array(list_x_elite))
        self.evaluator.eval(self.problem, off_elite, algorithm=self)

        # Track elite points for diagnostics
        if hasattr(self, 'exact_selected_tab'):
            self.exact_selected_tab.append(elite_pop.get("F"))
        if hasattr(self, 'exact_generated_tab'):
            self.exact_generated_tab.append(off_elite.get("F"))

        # --- Track 2: genetic operators on the rest ---
        n_genetic = self.init_pop_size - len(off_elite)
        if len(rest_pop) > 0 and n_genetic > 0:
            off_genetic = self.mating.do(self.problem, rest_pop,
                                         n_genetic, algorithm=self)
        else:
            off_genetic = Population.new("X", np.zeros((0, pb.n_var)))

        # --- Merge both tracks ---
        off = Population.merge(off_elite, off_genetic)

        if len(off) == 0:
            self.termination.force_termination = True
            return

        if len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    # ------------------------------------------------------------------
    # LEGACY / EXPERIMENTAL METHODS (kept as ablation variants)
    # ------------------------------------------------------------------

    @staticmethod
    def _custom_infill_0(self): # Code de base copié de _infill de pymoo.algorithms.base.genetic
        '''
        Do the mating using just the individuals with fitness=0 of the current population,
        even if it's a small number(?) (see _custom_infill_0a)
        A utiliser avec _temp_pop = '0'
        '''
        pop = None
        Ff = self.pop.get("Fit")
        if False: #Inutile de faire le tri(?) #Tech: Changer à False pour exécuter l'autre idée.
            FfSorted, fn1 = Advance4Util.get_sorted(Ff)
            pop = self.pop[FfSorted[fn1:]]
        else: #TODO #Inutile de faire le tri(?)
            Ffnul = np.flatnonzero(Ff==0) #TODO: Et si 0 n'existe pas!?
            assert len(Ffnul) > 0, "Pas d'individus avec fitness nulle. Corriger le code."
            pop = self.pop[Ffnul]

        off = self.mating.do(self.problem, pop, self.init_pop_size, algorithm=self)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings: #TODO: n_offsprings?
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    @staticmethod
    def _custom_infill_0a(self): # Code de base copié de _infill de pymoo.algorithms.base.genetic
        '''
        Do the mating using just the individuals with fitness=0 of the current population
        Number of points with fit==0 not <<
        A utiliser avec _temp_pop = '0'
        '''

        Ff = self.pop.get("Fit")
        FfSorted, fn1 = Advance4Util.get_sorted(Ff)
        d = len(FfSorted) - fn1
        if d < len(FfSorted)//2: #Différence with _infills0
            fn1 =  len(FfSorted)//2

        pop = self.pop[FfSorted[fn1:]]

        off = self.mating.do(self.problem, pop, self.init_pop_size, algorithm=self)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings: #TODO: n_offsprings?
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    @staticmethod
    def _custom_infill_1(self): # Code de base copié de _infill de pymoo.algorithms.base.genetic
        '''
        Using an exact method to generate a fixed number of new points from self.pop to be
        merged with the points that will be generated by mating from the rest of self.pop.
        Uses the last l points of the current population.
        '''

        l = 1 #len(self.pop)//2

        off1 = Advance4Util.do_exact_infill(self, self.pop[:l])

        if False:
            from pymoo.visualization.scatter import Scatter
            sc = Scatter(title=("Gen"))
            sc.add(self.pop[:l].get('F'), color="blue")
            sc.add(off1.get('F'), color="green")
            sc.do()
            sc.show()

        # do the mating using the rest of the current population
        off2 = self.mating.do(self.problem, self.pop[l:], len(self.pop[l:]), algorithm=self)

        off = Population.merge(off1, off2)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings: #TODO
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    @staticmethod
    def _custom_infill_1a(self): # Legacy version — use _custom_infill_paper for paper compliance
        '''
        Using an exact method to generate a fixed number of new points from fit==max to be merged with rest
        of the points generated by mating.

        Parameter: algorithm.tau (formerly self.IT) = number of elite individuals.
        Default tau=1 (set in _initialize_advance).
        '''
        assert len(self.pop) == self.init_pop_size, "len(self.pop) != self.init_pop_size"
        Ff = self.pop.get("Fit")

        off1 = None
        off2 = None
        # tau: number of elite individuals to apply exact infill (paper: τ)
        IT = getattr(self, 'tau', getattr(self, 'IT', 1))
        if True: # Select IT points with the highest fitness.
            FfSorted, fn1 = Advance4Util.get_sorted(Ff)
            l = len(FfSorted) - IT #l: index from which to take points for exact method
            self.exact_selected_tab.append(self.pop[FfSorted[l:]].get("F"))
            off1 = Advance4Util.do_exact_infill(self, self.pop[FfSorted[l:]], IT)
            self.exact_generated_tab.append(off1.get("F"))
            off2 = self.mating.do(self.problem, self.pop[FfSorted[:l]], l, algorithm=self, random_state=self.random_state)

        off = Population.merge(off1, off2)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings: #TODO
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    @staticmethod #Pas intéressante
    def _custom_infill_1aa(self): # Code de base copié de _infill de pymoo.algorithms.base.genetic
        '''
        Using an exact method to generate a fixed number of new points from fit==0 to be merged with rest
        of the points generated by mating.
        '''
        algorithm = self
        off = None

        if (algorithm.n_iter % 90 == 0):
            Ff = algorithm.pop.get("Fit")
            FfSorted, fn1 = Advance4Util.get_sorted(Ff)

            l = len(FfSorted) - 2

            off1 = Advance4Util.do_exact_infill(algorithm, algorithm.pop[FfSorted[l:]])

            off2 = algorithm.mating.do(algorithm.problem, algorithm.pop[FfSorted[:l]], l, algorithm=algorithm)

            off = Population.merge(off1, off2)
        else:
            off = algorithm.mating.do(algorithm.problem, algorithm.pop, algorithm.n_offsprings, algorithm=algorithm)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            algorithm.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < algorithm.n_offsprings: #TODO
            if algorithm.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    @staticmethod
    def _custom_infill_1b(self): # Code de base copié de _infill de pymoo.algorithms.base.genetic
        '''
        Using an exact method to generate a fixed number of new points from worst fit to be merged with rest
        of the points generated by mating.
        '''

        Ff = self.pop.get("Fit")
        FfSorted, fn1 = Advance4Util.get_sorted(Ff)

        l = 10

        off1 = Advance4Util.do_exact_infill(self, self.pop[FfSorted[:l]])

        off2 = self.mating.do(self.problem, self.pop[FfSorted[l:]], len(FfSorted)-l, algorithm=self)

        off = Population.merge(off1, off2)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings: #TODO
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off
