import numpy as np

from ibeas.ibea_t import IBEATest
from ibea_stats import IBEAStats, show_stats, do_graph, show_perf_graph, do_iter_graph

# TODO: Not used yet
class AlgorithmParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# See ibea_advance_4.py
class IBEATest4:

    @staticmethod
    def construct_filename(NIter, POP, algorithm, problem, step):
        return f"generated/nibea{problem.name()}({problem.n_var},{problem.n_obj})[temp_pop{algorithm.temp_pop}][sel_pop{algorithm.sel_pop}][env{algorithm.env_sel_method}][sc_method{algorithm.sc_method}]({NIter},{POP}){algorithm.ibea_advance.adv_name()}[step{step}]"

    @staticmethod
    def callback(res, filename, step):
          from ibea_recorder import IBEARecord
          IBEARecord.record_video(res, filename=filename+'.mp4', step=step)
          #IBEARecord.record_video_1(res, filename=filename+'-1.mp4')
          #IBEARecord.record_video_2(res, g=1, step=10, filename=filename+'-2.mp4')

    # Appel de la méthode et affichage des stats
    @staticmethod
    def test(NIter, POP, problem, ref_point=None, K=None, **kwargs):

        if K is None:
            K = 1

        res, algorithm, n_gen = IBEATest.ibea_4(problem=problem, K=K, pop_size=POP, n_gen=NIter, **kwargs)

        #
        if 'cb' in kwargs:
            callback = kwargs['cb']
            if callback is not None:
                stp = 1
                filename = IBEATest4.construct_filename(NIter, POP, algorithm, problem, stp)
                callback(res, filename, stp) #Without the extension
        #
        #if 'do_g' in kwargs and kwargs.get('do_g'):
        if False:
            cb_graph = kwargs.get('do_g')
            #if cb_graph:
            #
            #do_iter_graph(problem, res, 0, "NIBEA")
            #do_iter_graph(problem, res, 1, "NIBEA")
            #do_iter_graph(problem, res, n_gen-3, "NIBEA")
            do_iter_graph(problem, res, n_gen-2, "NIBEA")
            #
            do_graph(problem, res, "NIBEA")
            #
            #IBEAStats.stats_1(res)
            #IBEAStats.stats_2a(res, n_gen, 20)
            #IBEAStats.stats_3(res, n_gen, 16)
            #
            max_hv, min_igd = show_perf_graph(problem, res, "NIBEA")
            #do_iter_graph(problem, res, max_hv, "NIBEA")
            #do_iter_graph(problem, res, min_igd, "NIBEA")

            #Exact method points visualization
            from pymoo.visualization.scatter import Scatter
            plot = Scatter()
            plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
            plot.add(np.array(algorithm.exact_generated_tab[-10:]), facecolor="none", edgecolor="red")
            plot.add(np.array(algorithm.exact_selected_tab[-10:]), facecolor="none", edgecolor="blue")
            plot.show()

        print("algorithm.count_sel_pop = ", algorithm.count_sel_pop)
        #print("Exact points: ", algorithm.exact_generated_tab)

        if ref_point is None:
            ref_point = problem.nadir_point()

        return show_stats(problem, ref_point, res, "NIBEA")
        #plt.show()
        #

    @staticmethod
    def test_zdt1():
        from pymoo.problems import get_problem
        #method0 pour le départage.
        #Converge. rank=3.
        #IBEATest4.test(100, 100, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=3. K>> n'améliore pas et crée des trous dans la FP.
        #IBEATest4.test(100, 100, get_problem("zdt1"), np.array([1, 1]), K=1000)
        #Converge. rank=3. Mieux qu'avant. Belle FP.
        #IBEATest4.test(100, 200, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=2. Mieux qu'avant. Belle FP.
        #IBEATest4.test(100, 300, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=2. Mieux qu'avant. Belle FP.
        #IBEATest4.test(100, 400, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=2. Mieux qu'avant? Belle FP.
        #IBEATest4.test(100, 500, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=3. Trous dans la FP
        #IBEATest4.test(200, 200, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=3. Encore plus de trous dans la FP. Le plus mauvais jusqà présent.
        #IBEATest4.test(300, 200, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=2. 50 itérations mieux que 100 en terme de rank !
        #IBEATest4.test(50, 200, get_problem("zdt1"), np.array([1, 1]))

        #method5 pour le départage. Prends plus de temps que method0.
        #Converge. rank=3.
        #IBEATest4.test(100, 100, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=2. Mieux qu'avant. Belle FP. Meill rank que method0.
        #IBEATest4.test(100, 200, get_problem("zdt1"), np.array([1, 1]))

        #method2 pour le départage.
        #Converge. rank=3.
        #IBEATest4.test(100, 100, get_problem("zdt1"), np.array([1, 1]))
        #Converge. rank=2. Mieux qu'avant.
        #IBEATest4.test(100, 200, get_problem("zdt1"), np.array([1, 1]))

        #method2a pour le départage.
        #Meill que method2. Converge. rank=2.
        IBEATest4.test(100, 200, get_problem("zdt1"), np.array([1, 1]))

        #method2aa pour le départage.
        #
        #IBEATest4.test(100, 100, get_problem("zdt1"), np.array([1, 1]))

    @staticmethod
    def test_zdt2():
        from pymoo.problems import get_problem
        #method2a pour le départage.
        #?
        #IBEATest4.test(100, 100, get_problem("zdt2"), np.array([1, 1]))
        #Converge. rank=2. Mieux qu'avant.
        #IBEATest4.test(100, 200, get_problem("zdt2"), np.array([1, 1]))
        #Converge. rank=3.
        IBEATest4.test(200, 100, get_problem("zdt2"), np.array([1, 1]))

    @staticmethod
    def test_zdt3():
        from pymoo.problems import get_problem
        #method2a pour le départage.
        #converge. rank 2 en HV, 3 en IGD.
        #IBEATest4.test(100, 100, get_problem("zdt3"), np.array([1, 1]))
        #converge. rank 2.
        IBEATest4.test(100, 200, get_problem("zdt3"), np.array([1, 1]))

    @staticmethod
    def test_zdt4(params=None):
        print('''
        *****************
        NIBEA Algorithm (inside IBEATest4.test_zdt4())
        ZDT4 Problem
        *****************
        ''')
        from pymoo.problems import get_problem
        if params is not None:
            pass #TODO
        #method2a pour le départage.
        #Pas bonne.
        #IBEATest4.test(100, 100, get_problem("zdt4"), np.array([1, 1]))
        #Meill en HV que NSGA2 et SPEA2
        #IBEATest4.test(100, 300, get_problem("zdt4"), np.array([1, 1]), K=100)
        #With callback
        #IBEATest4.test(200, 100, get_problem("zdt4"), np.array([1, 1]), K=1, callback=IBEATest4.callback)
        #Now testing:
        IBEATest4.test(20, 10, get_problem("zdt4"), np.array([1, 1]), K=1)

        #method2aa pour le départage.
        #Pas bonne.
        #IBEATest4.test(100, 100, get_problem("zdt4"), np.array([1, 1]))

    @staticmethod
    def main(params=None):
        print('''
        *****************
        NIBEA Algorithm (inside IBEATest4.main())
        *****************
        ''')
        from pymoo.problems import get_problem
        if params is not None:
            #params.problem_name? params.?
            pass #TODO
            return AlgorithmParams(**params)
        #Bons résultats pour PROB1
        #IBEATest4.test() # 4
        #Bons résultats pour bnh
        #IBEATest4.test(100, 100, get_problem("bnh"), np.array([140, 55])) # 4
        #En général, mauvais résultats pour osy
        #IBEATest4.test(100, 200, get_problem("osy"), np.array([-50, 90]), K=100) # 4
        #Mauvais résultats pour tnk
        #IBEATest4.test(12, 100, get_problem("tnk"), np.array([1.1, 1.1])) # 4
        #Bons résultats pour truss2d
        #IBEATest4.test(12, 100, get_problem("truss2d"), np.array([0.06, 100000])) # 4
        #??? résultats pour welded_beam. Meill que SPEA2 pour Iter=12, 100. Rés bizarre pour Iter=50
        #IBEATest4.test(12, 100, get_problem("welded_beam"), np.array([40, 0.016]), K=100) # 4
        #IBEATest4.test_zdt1()
        #IBEATest4.test_zdt2()
        #IBEATest4.test_zdt3()
        IBEATest4.test_zdt4()
        #Bons résultats pour zdt5
        #IBEATest4.test(100, 100, get_problem("zdt5"), np.array([35, 10])) # 4

        #Bons résultats pour zdt6 (mais avec It=200, K=4)
        #IBEATest4.test(120, 300, get_problem("zdt6"), np.array([1, 4]), K=100) # 4

        #Bons résultats pour mw1
        #IBEATest4.test(100, 100, get_problem("mw1"), np.array([1, 1])) # 4
        #Bons résultats pour mw2
        #IBEATest4.test(100, 100, get_problem("mw2"), np.array([1, 1])) # 4
        #Bons résultats pour mw3
        #IBEATest4.test(100, 100, get_problem("mw3"), np.array([1, 1])) # 4
        #Bons résultats pour mw5
        #IBEATest4.test(100, 100, get_problem("mw5"), np.array([1, 1])) # 4
        #Bons résultats pour mw6
        #IBEATest4.test(100, 100, get_problem("mw6"), np.array([1, 1])) # 4
        #Bons résultats pour mw7
        #IBEATest4.test(100, 100, get_problem("mw7"), np.array([1.2, 1.2])) # 4
        #Mauvais résultats pour mw9, (100,100)
        #IBEATest4.test(200, 200, get_problem("mw9"), np.array([1, 1]), K=100) # 4
        #Mauvais résultats pour mw10
        #IBEATest4.test(12, 10, get_problem("mw10"), np.array([1, 1.2])) # 4


if __name__ == "__main__":
    from pymoo.problems import get_problem
    pb = get_problem("zdt1")
    algo_params = {"inf": '1a', "temp_pop": '0', "env_sel_method": '1', "sel_pop": '0', "sc_method": '0'}
    test_params = {"do_g": True}#, "cb": IBEATest4.callback}
    IBEATest4.test(100, 100, pb, pb.nadir_point(), K=1, **test_params, **algo_params)

