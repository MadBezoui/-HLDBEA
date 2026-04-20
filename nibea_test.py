from pymoo.config import Config
Config.warnings['not_compiled'] = False

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

import sys
sys.path.append('./src')

from ibea_recorder import IBEARecord
from ibea_callbacks import MyCallback
from ibea import IBEA
from ibeas.nibea.ibea_test_4 import IBEATest4
from ibeas.ibea_t import IBEATest
from ibeas.nibea.data_analysis import prepare_data, calculate_summary_stats, perform_wilcoxon_tests
from ibeas.nibea.algo_execution import TestAlgorithms
from ibea_stats import plot_median_run, save_boxplots, save_convergence_curves

from ibea_recorder import IBEARecord


global_verbose = False

RESULTS_OUTPUT_DIR="generated"

def test_pb_by_algos(NIter=100, NPop=100, problem=None, algos=None, ref_point=None, **kwargs):
    """
    Tests multiple algorithms on a given problem and returns their results.

    Parameters:
    ----------
    NIter : int, optional
        Number of iterations for the algorithms (default: 100).
    NPop : int, optional
        Size of the population for the algorithms (default: 100).
    problem : pymoo problem object, optional
        The problem to be optimized.
    algos : dict, optional
        Dictionary with algorithm names as keys and boolean flags as values (default: None).
        Supported algorithms: 'NIBEA+', 'NIBEA', 'NSGA2', 'SPEA2', 'SMSEMOA'
    ref_point : list, optional
        Reference point for hypervolume calculation (default: problem's nadir point).

    Returns:
    -------
    results : list
        List of dictionaries containing statistics (e.g., HV, GD, IGD) for each algorithm.
    """
    if global_verbose:
        print("*** PROBLEM INFOS: ***\n", problem)
    else:
        print(f"*** PROBLEM : {problem.name()} ***")

    if ref_point is None:
        if problem.n_obj > 3:
            #TODO ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
            ref_point = problem.nadir_point(ref_dirs=ref_dirs)
        else:
            ref_point = problem.nadir_point()

    res = []
    if algos["NIBEA"]:
        if kwargs.get('do_v'):
            if problem.n_obj < 4:
                a = IBEATest4.test(NIter, NPop, problem, ref_point, K=1, **kwargs, cb=IBEATest4.callback)
            else:
                a = IBEATest4.test(NIter, NPop, problem, ref_point, K=1, **kwargs)
        else:
            a = IBEATest4.test(NIter, NPop, problem, ref_point, K=1, **kwargs)
        res.append(a)
    if algos["NSGA2"]:
        a = TestAlgorithms.test_nsga2(NIter, NPop, problem, **kwargs)
        res.append(a)
    if algos["SPEA2"]:
        a = TestAlgorithms.test_spea2(NIter, NPop, problem, **kwargs)
        res.append(a)
    if algos["NSGA3"]:
        a = TestAlgorithms.test_nsga3(NIter, NPop, problem, **kwargs)
        res.append(a)
    if algos["RVEA"]:
        a = TestAlgorithms.test_rvea(NIter, NPop, problem, **kwargs)
        res.append(a)
    if algos["MOEAD"]:
        a = TestAlgorithms.test_moead(NIter, NPop, problem, **kwargs)
        res.append(a)
    if algos["AGEMOEA"]:
        a = TestAlgorithms.test_agemoea(NIter, NPop, problem, **kwargs)
        res.append(a)
    #a = TestAlgorithms.test_smsemoa(NIter, NPop, problem, **kwargs)#, cb=IBEARecord.record_video)
    return res

def run_and_collect(NIter=100, NPop=100, problem=None, sp=2, **kwargs):
    """
    Runs multiple algorithms on a given problem for a specified number of samples and collects their results.

    Parameters:
    ----------
    NIter : int, optional
        Number of iterations for the algorithms (default: 100).
    NPop : int, optional
        Size of the population for the algorithms (default: 100).
    problem : pymoo problem object, optional
        The problem to be optimized.
    sp : int, optional
        Number of samples (i.e., independent runs) to collect results for (default: 5).

    Returns:
    -------
    results : dict
        Dictionary with algorithm names as keys and dictionaries of metric values as values.
        Metric dictionaries have metric names (e.g., 'HV', 'GD', 'IGD') as keys and lists of sampled values as values.
    """
    algorithms = kwargs.get('algorithms')
    if not algorithms:
        algorithms = {"NIBEA":True, "NSGA2":True, "SPEA2":True, "NSGA3":True, "RVEA":True, "MOEAD":True, "AGEMOEA":True}#, "SMSEMOA":False}
    metrics = ["HV", "GD", "IGD","IGDPlus"]
    results = {alg: {metric: [] for metric in metrics} for alg, val in algorithms.items() if val}
    res_by_algos = {alg: [] for alg, val in algorithms.items() if val}

    for s in range(sp):
        print(f"Run n° {s+1:{2}} / {sp}")
        algo_res = test_pb_by_algos(NIter, NPop, problem, algorithms, seed=s, **kwargs)
        #TODO: if there is something to do after each iteration, do it here.
        i = 0
        for alg, _ in results.items():
            results[alg]["HV"].append(algo_res[i][0])
            results[alg]["GD"].append(algo_res[i][1])
            results[alg]["IGD"].append(algo_res[i][2])
            results[alg]["IGDPlus"].append(algo_res[i][3])
            res_by_algos[alg].append(algo_res[i][4])
            i += 1

    if 'do_g' in kwargs and kwargs.get('do_g'):
        print("GENERATING AND SAVING FIGURES...")
        save_boxplots(problem, results, kwargs.get('exp_name', ''), f"{RESULTS_OUTPUT_DIR}")
        #TODO Attention, Trop gourmand en ressources!
        #save_convergence_curves(problem, res_by_algos, kwargs.get('exp_name', ''), f"{RESULTS_OUTPUT_DIR}")
        for alg, _ in results.items():
            plot_median_run(res_by_algos[alg], results[alg]["HV"], problem, alg, kwargs.get('exp_name', ''), f"{RESULTS_OUTPUT_DIR}")

    return results

def do_algos_stats(probs_config, n_samp=2, Probs=None, **kwargs):
    """
    Orchestrates the comparison of multiple algorithms on a set of problems, collecting and analyzing their results.

    Parameters:
    ----------
    NIter : int, optional
        Number of iterations for the algorithms (default: 100).
    NPop : int, optional
        Size of the population for the algorithms (default: 100).
    n_samp : int, optional
        Number of samples (i.e., independent runs) to collect results for (default: 2).
    Probs : list, optional
        List of problem names to include in the comparison (default: a predefined list of problems).

    Returns:
    -------
    None
    """
    if Probs is None:
        #Error: you must give the problems...
        return
    else:
        ALL_TEST_PROBS = Probs


    print("Test problems:", str(ALL_TEST_PROBS))
    #[print(pb_name) for pb_name in ALL_TEST_PROBS]

    if len(ALL_TEST_PROBS) > 1 and isinstance(probs_config, tuple):
        probs_config = [probs_config] * len(ALL_TEST_PROBS)

    data = {str(pb_name[0]): run_and_collect(*probs_config[i], get_problem(*pb_name if isinstance(pb_name, tuple) else (pb_name,)), n_samp, **kwargs) for i, pb_name in enumerate(ALL_TEST_PROBS)}

    # Convert the list of dictionaries to a DataFrame
    df = prepare_data(data)

    # Display the summary statistics
    #print(df)
    #df.to_csv(f"output/results.csv", index=False, header=False)

    if n_samp >= 2:
        wr = perform_wilcoxon_tests(data, ALL_TEST_PROBS, **kwargs)

        calculate_summary_stats(df, wr)
    #print(f"POP:{NPop}\nITER:{NIter}\nSAMPLE:{n_samp}")
    print(f"{probs_config}\nSAMPLE:{n_samp}")
    print(f"**kwargs: {kwargs}")







import argparse
import yaml
import os

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run LDBEA Optimization Experiments")
    
    # Arguments for the three config types
    parser.add_argument("--profile", type=str, default="configs/profile_debug.yaml", help="Path to profile yaml")
    parser.add_argument("--params", type=str, default="configs/params_standard.yaml", help="Path to params yaml")
    parser.add_argument("--algos", type=str, default="configs/algos.yaml", help="Path to algos yaml")
    parser.add_argument("--out", type=str, default="results", help="Suffix for output files")

    args = parser.parse_args()

    # 1. Load Configs
    profile = load_yaml(args.profile)
    params_raw = load_yaml(args.params)
    algos = load_yaml(args.algos)

    # 2. Flatten Params for the function
    # Merges fitness, infill, and env_selection into one dict
    algo_params = {**params_raw['fitness'], **params_raw['infill'], **params_raw['env_selection']}
    
    # 3. Prepare Problem Config
    num_probs = len(profile['probs'])
    p_config = profile['probs_config']

    test_params = {"stop_criterion": profile['stop_criterion'], "do_g": True, "save_history": True}

    # 4. Execute
    print(f"Launching Experiment: {profile['name']} for {profile['n_samp']} runs...")
    do_algos_stats(
        probs_config=[tuple(p) for p in p_config],
        n_samp=profile['n_samp'],
        Probs=[tuple(p) for p in profile['probs']],
        algorithms=algos,
        exp_name=profile["name"], #args.out, # Ensure do_algos_stats uses this for naming files
        **test_params,
        **algo_params
    )

if __name__ == "__main__":
    main()
