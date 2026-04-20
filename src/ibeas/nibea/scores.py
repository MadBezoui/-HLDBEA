import numpy as np

from util.misc import get_caller_function

class CalcScore:

    @staticmethod #ndim
    def get_dom_points(i, F):
        """
        Get The dominating points of i.
        A point j dominates i if f_j(k) <= f_i(k) for all k, with strict inequality on at least one.
        Here we use the <= convention on the first axis; the mask then filters the rest.
        """
        #assert get_caller_function() == 'calc_all_scores', f"get_dom_points() called elsewhere ({get_caller_function()})."

        v = (F[:, 0] <= F[i,0]) # Points at the same level on the first axis are also counted.
        v[i] = False
        Fv = F[v]
        #
        # Initialize the filter mask with all True values
        mask = np.ones(Fv.shape[0], dtype=bool)
        # Deduce the dimensions from the shape of F
        dims = list(range(Fv.shape[1]))
        dims.remove(0)
        for dim in dims:
            # Update the mask to include only points within the bounds
            mask &= (Fv[:, dim] <= F[i, dim])
        Fv = Fv[mask]

        return Fv

    @staticmethod
    def calc_score_axis(i, F, Fv, pas, ax, K):
        '''
        Returns the list of points that dominate i in the neighborhood along the ax axis.
        Fv contains all the dominating points of i; here we select just those along the ax axis.
        Fv as a param to not call get_dom_points() for each axis.

        Paper reference: Definition 3 — Vₐ(A) = { B ∈ Dom(A) : fₐ(A) − pₐ ≤ fₐ(B) ≤ fₐ(A) }
        where pₐ = (max fₐ − min fₐ) / N  (computed upstream in calc_scores()).
        '''
        if len(Fv)==0:
            return []

        a = F[i, ax] - pas
        v = (Fv[:, ax] >= a)
        Fv = Fv[v]

        return Fv


    @classmethod
    def calc_score(cls, i, F, Fv, steps, K):
        '''
        Calculates the score of i.
        Paper: score(A) = Σₐ |Vₐ(A)|
        Returns a list of the number of dominating points of i along each axis.
        Be aware that some points will be used more times in the global score.
        '''
        ls = []
        axis = list(range(F.shape[1]))
        for ax in axis:
            # Get the F of all the points dominating i along the ax axis in its neighborhood
            ll = cls.calc_score_axis(i, F, Fv, steps[ax], ax, K)
            # Calculates the score along the ax axis as the nb of dominating points of i
            ls.append(len(ll))

        return ls

    @classmethod #Incubation #Pas intéressante, non utilisé.
    def calc_score_1(cls, i, F, Fv, steps, K):
        '''
        Calculates the score of i.
        Uses the distance between... along each axis (dimension). Use for example the sum
        of all the distances along each axis...
        Autres idées à implémenter inch'Allah.
        - Si le point min selon un axe ax est dans une certain distance -> +1 au score.
        '''
        ls = []
        axis = list(range(F.shape[1]))
        for ax in axis:
            Fax = cls.calc_score_axis(i, F, Fv, steps[ax], ax, K)
            ds = list(range(F.shape[1]))
            ds.remove(ax)
            #[Fax[p, ax] for p in np.arange(len(Fax)) for d in ds]
            w = 0
            if len(Fax): #If Fax is not empty
                #w = np.sum(np.exp(Fax[:, ds] - F[i, ds]))
                w = np.prod(np.exp(Fax[:, ds] - F[i, ds]))
            ls.append(len(Fax) + w)

        return ls

    @classmethod
    def calc_score_2(cls, i, F, Fv, steps, K):
        '''
        Calculates the score of an individual i.
        Uses the number of individuals dominating i in its neighborhood
        and the maximal distance between...
        Fv: all dominating points of i.
        steps:...
        '''
        ls = []
        axis = list(range(F.shape[1]))
        for ax in axis:
            # Get all the points dominating i along the ax axis in its neighborhood
            Fax = cls.calc_score_axis(i, F, Fv, steps[ax], ax, K)

            if len(Fax)==0:
                return 0

            # Get the closest individual to i among its neighborhood along the ax axis.
            fmax = np.max(Fax[:, ax])
            w =  F[i, ax] - fmax
            #Idée 1. See calc_score_2a for other ideas.
            ls.append(len(Fax) + np.exp(-w))

        return ls

    @classmethod
    def calc_score_2a(cls, i, F, Fv, steps, K):
        '''
        Same as calc_score_2 except for the calculation of the closest point to i
        which uses ALL the dominating points instead of those in the neighborhood.
        '''
        ls = []
        axis = list(range(F.shape[1]))
        for ax in axis:
            # Get all the points dominating i along the ax axis in its neighborhood
            Fax = cls.calc_score_axis(i, F, Fv, steps[ax], ax, K)

            if len(Fax)==0:
                return 0

            # Get the closest individual to i among ALL its dominating points (diff with calc_score_2) along the ax axis.
            fmax = np.max(Fv[:, ax]) #Fv instead of Fax like in calc_score_2
            w =  F[i, ax] - fmax
            #Idée 1. See calc_score_2a for other ideas.
            ls.append(len(Fax) + np.exp(-w))

        return ls

    @classmethod
    def calc_score_2b(cls, i, F, Fv, steps, K):
        '''
        Same as calc_score_2...
        Calculates the score of i.
        Uses the distance between...
        Fv: all dominating points of i.
        steps:...
        '''
        ls = []
        axis = list(range(F.shape[1]))
        for ax in axis:
            # Get all the points dominating i along the ax axis in its neighborhood
            Fax = cls.calc_score_axis(i, F, Fv, steps[ax], ax, K)

            if len(Fax)==0:
                return 0

            fmax = np.max(Fax[:, ax])
            w =  F[i, ax] - fmax
            #Idée 2. See calc_score_2 for idea 1.
            #ls.append(w)
            #Idée 3
            ls.append(w - np.exp(-len(Fax)))

        return ls

    @classmethod
    def calc_score_ibea(cls, i, F, Fv, steps, K): #TODO Not used yet
        '''
        Calculates the score of i using zitzler2004 just on the neighborhood of i (not all the pop).
        Return ...
        Le problème est que pour un point non dominé, le voisinage est vide, val(I)=? (0?)
        '''
        from ibea_util import IBEAFfitOriginal, IBEAFfit

        ls = []
        axis = list(range(F.shape[1]))
        for ax in axis:
            # Get the F of all the points dominating i along the ax axis in its neighborhood
            _F = cls.calc_score_axis(i, F, Fv, steps[ax], ax, K)
            # Calculates the score along the ax axis
            F_i1 = [np.exp((-1)*(np.max(_F[i2] - F[i]))/0.05) if len(_F) > 0 else 0 for i2 in np.arange(len(_F))]

            ls.append( np.sum( F_i1 ) )

        return ls

    @classmethod
    def calc_all_scores(cls, F, steps, algorithm):
        """
        Dispatches to the appropriate scoring aggregation method.

        The default (sc_method == "0") implements the formula from the paper:
            fit(A) = -score(A) - λ · dist(A)
        where score(A) = Σₐ |Vₐ(A)|  and
              dist(A)  = ‖f(A) - z*‖₂ / d_max  (normalised distance to the ideal point).
        λ is read from algorithm.lmbda (default 0.1, as per paper Section 3.2).
        """
        #Getting The dominating points of all points once. To be used for all axis.
        Fvs = [cls.get_dom_points(i1, F) for i1 in np.arange(len(F))]
        scores = [cls.calc_score(i1, F, Fvs[i1], steps, algorithm.K) for i1 in np.arange(len(F))]

        sc_method = algorithm.sc_method
        if (sc_method == "0"):
            return cls.calc_all_scores_0(F, scores, steps, algorithm)
        elif sc_method == "1": #TODO: incub
            return cls.calc_all_scores_1(F, scores, steps)
        elif sc_method == "3":
            return cls.calc_all_scores_3(F, scores, steps)
        elif sc_method == "4":
            return cls.calc_all_scores_4(F, Fvs, scores, steps)
        elif sc_method == "a":
            #Utiliser calc_score_2 à la place de calc_score
            scores_a = [cls.calc_score_2(i1, F, Fvs[i1], steps, algorithm.K) for i1 in np.arange(len(F))]
            return cls.calc_all_scores_0(F, scores_a, steps, algorithm)
        elif sc_method == "2a":
            #Utiliser calc_score_2a à la place de calc_score
            scores_a = [cls.calc_score_2a(i1, F, Fvs[i1], steps, algorithm.K) for i1 in np.arange(len(F))]
            return cls.calc_all_scores_0(F, scores_a, steps, algorithm)
        elif sc_method == "2b":
            #Utiliser calc_score_2b à la place de calc_score
            scores_a = [cls.calc_score_2b(i1, F, Fvs[i1], steps, algorithm.K) for i1 in np.arange(len(F))]
            return cls.calc_all_scores_0(F, scores_a, steps, algorithm)
        elif sc_method == "ibea":
            #Utiliser calc_score_ibea à la place de calc_score
            scores_a = [cls.calc_score_ibea(i1, F, Fvs[i1], steps, algorithm.K) for i1 in np.arange(len(F))]
            return cls.calc_all_scores_0(F, scores_a, steps, algorithm)
        else: #"2" by default (experimental variant: adds product term):
            return cls.calc_all_scores_2(F, scores, steps)


    @classmethod
    def calc_all_scores_0(cls, F, scores, steps, algorithm=None):
        '''
        PAPER-CONFORMANT fitness formula (Algorithm 1, Section 3.2):
            fit(A) = -score(A) - λ · dist(A)

        where:
            score(A) = Σₐ |Vₐ(A)|        (sum of dominating neighbours per axis)
            dist(A)  = ‖f(A) - z*‖₂ / d_max   (normalised L2 distance to the ideal point z*)
            λ        = algorithm.lmbda    (default 0.1, paper Section 3.2)

        If algorithm is None (e.g., called from legacy code), falls back to -Σ(score).
        '''
        base_scores = np.array([(-1) * np.sum(ll) for ll in scores])

        if algorithm is None:
            return base_scores

        lmbda = getattr(algorithm, 'lmbda', 0.1)

        if lmbda == 0.0:
            return base_scores

        # z*: ideal point — component-wise minimum over the population
        z_star = np.min(F, axis=0)      # shape (n_obj,)

        # Raw L2 distances to ideal point
        dists = np.linalg.norm(F - z_star, axis=1)   # shape (N,)

        # Normalise by the maximum distance (avoid div-by-zero)
        d_max = np.max(dists)
        if d_max > 0:
            dists = dists / d_max

        return base_scores - lmbda * dists

    @classmethod
    def calc_all_scores_2(cls, F, scores, steps):
        '''
        Experimental variant: adds a product term to penalise uneven per-axis scores.
        fit(A) = -Σ(score) - Π(score)
        Kept as an alternative but NOT the paper default (use sc_method="0" for paper compliance).
        '''
        scores = [(-1)*np.sum(ll)+(-1)*np.prod(ll) for ll in scores]
        return np.array(scores)

    @classmethod
    def calc_all_scores_3(cls, F, scores, steps):
        '''
        Calculates the score of all the individuals in the population by summing and product...
        '''
        scores = [(-1)*np.sum(ll)*np.minimum(1, np.prod(ll)) for ll in scores]
        return np.array(scores)

    @classmethod #TODO:13/06/2025
    def calc_all_scores_4(cls, F, Fvs, scores, steps):
        '''
        Calculates the score of all the individuals in the population...
        '''
        #Bonne
        scores = [(-1)*np.sum(ll)-np.exp(len(Fvs[i])-np.sum(ll)) for i, ll in enumerate(scores)]
        #
        #scores = [(-1)*np.sum(ll)-np.exp(-(1+np.sum(ll))/(1+len(Fvs[i]))) for i, ll in enumerate(scores)]
        return np.array(scores)

    @classmethod #TODO: incub
    def calc_all_scores_1(cls, F, scores, steps):
        '''
        Calculates the score of all the individuals in the population.
        '''
        for s in scores: #by point
            if np.sum(s) != 0: #som dé scor sr tt lé dim
                for i in range(len(s)):  #by dim: lx, ly, lz,...
                    if s[i] == 0:
                        pass
                        #TODO: distance to nadir point in the ax where == 0
                        np.max(F[:,])
        scores = [(-1)*np.sum(ll) for ll in scores]
        return np.array(scores)
