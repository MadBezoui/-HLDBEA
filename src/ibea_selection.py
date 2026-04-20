import numpy as np

from pymoo.operators.selection.tournament import compare

#To be removed. Replaced with nibea_binary_tournament
def ibea_binary_tournament_4(pop, P, algorithm, **kwargs):
    return nibea_binary_tournament(pop, P, algorithm, **kwargs)

def nibea_binary_tournament(pop, P, algorithm, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    S = np.full(n_tournaments, -1) #, dtype=np.int)

    def idea1():
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
            a_fit, b_fit = pop[a].get("Fit"), pop[b].get("Fit")

            # if at least one solution is infeasible
            if a_cv > 0.0 or b_cv > 0.0:
                S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True, random_state=algorithm.random_state)

            # both solutions are feasible
            else:

                if  a_fit > b_fit:
                    S[i] = a

                elif a_fit < b_fit:
                    S[i] = b

                else:
                    #TODO
                    S[i] = b
                #    cd_a, cd_b = pop[a].get("crowding"), pop[b].get("crowding")
                #    if cd_a and cd_b:
                        #print("CROWD: ", cd_a, cd_b)
                #        if cd_a > cd_b:
                #            S[i] = a

        return S

    def idea2():
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
            a_fit, b_fit = pop[a].get("Fit"), pop[b].get("Fit")

            # if at least one solution is infeasible
            if a_cv > 0.0 or b_cv > 0.0:
                S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True, random_state=algorithm.random_state)

            # both solutions are feasible
            else:

                if  a_fit > b_fit:
                    S[i] = a

                elif a_fit < b_fit:
                    S[i] = b

                else:
                    from pymoo.util.dominator import Dominator
                    #Idea: utiliser la dominance
                    dm = Dominator.get_relation(pop[a].get("F"), pop[b].get("F"))
                    if dm == 1:
                        S[i] = a
                    elif dm == -1:
                        S[i] = b
                    #elif pop[a].get("FitLife") > pop[b].get("FitLife"):
                    #    S[i] = a
                    else:
                        S[i] = b

        return S

    # The same as dom_first_binary_tournament below
    def idea3(tech1=False):
        '''
        tech1: update fitness score if a point is dominated.
        '''
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
            a_fit, b_fit = pop[a].get("Fit"), pop[b].get("Fit")

            # if at least one solution is infeasible
            if a_cv > 0.0 or b_cv > 0.0:
                S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True, random_state=algorithm.random_state)

            # both solutions are feasible
            else:
                from pymoo.util.dominator import Dominator
                #Idea: utiliser la dominance
                dm = Dominator.get_relation(pop[a].get("F"), pop[b].get("F"))
                if dm == 1:
                    S[i] = a
                    #tech1
                    if tech1:
                        if b_fit == 0:
                          pop[b].set("Fit", -1)
                          pop[b].set("MinFit", -1)
                elif dm == -1:
                    S[i] = b
                    #tech1
                    if tech1:
                        if a_fit == 0:
                          pop[a].set("Fit", -1)
                          pop[a].set("MinFit", -1)
                else:
                    if  a_fit > b_fit:
                        S[i] = a

                    elif a_fit < b_fit:
                        S[i] = b

                    else:
                         S[i] = b

        return S

    #return idea1()
    return idea2()
    #return idea3(True)


def dom_first_binary_tournament(pop, P, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    S = np.full(n_tournaments, -1) #, dtype=np.int)

    from pymoo.util.dominator import Dominator

    def incub_1():
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            #Idea: utiliser la dominance
            dm = Dominator.get_relation(pop[a].get("F"), pop[b].get("F"))
            if dm == 1:
                S[i] = a
            elif dm == -1:
                S[i] = b
            else:

                # if the first individual is better, choose it
                if pop[a].get("Fit") > pop[b].get("Fit"):
                    S[i] = a

                # otherwise take the other individual
                else:
                    S[i] = b

        return S

    return incub_1()


def ibea_binary_tournament(pop, P, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    S = np.full(n_tournaments, -1) #, dtype=np.int)

    from pymoo.util.dominator import Dominator

    def incub_2():
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            # if the first individual is better, choose it
            if pop[a].get("Fit") > pop[b].get("Fit"):
                S[i] = a

            # otherwise take the other individual
            elif pop[a].get("Fit") < pop[b].get("Fit"):
                S[i] = b

            else:
                dm = Dominator.get_relation(pop[a].get("F"), pop[b].get("F"))
                if dm == 1:
                    S[i] = a
                elif dm == -1:
                    S[i] = b
                else:
                    S[i] = a

        return S

    def original():
        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            # if the first individual is better, choose it
            if pop[a].get("Fit") > pop[b].get("Fit"):
                S[i] = a

            # otherwise take the other individual
            else:
                S[i] = b

        return S

    #return incub_2()
    return original()
