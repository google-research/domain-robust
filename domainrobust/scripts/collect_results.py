# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import argparse
import os
import sys

import numpy as np
from domainrobust import datasets
from domainrobust import algorithms
from domainrobust.lib import misc, reporting
from domainrobust import model_selection
from domainrobust.lib.query import Q

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    elif np.array(data).shape[1]==1:
        mean = 100 * np.mean(list(data))
        err = 100 * np.std(list(data) / np.sqrt(len(data)))
        if latex:
            return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
        else:
            return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)
    elif np.array(data).shape[1]==2:
        nat_acc = np.array(data)[:,0]
        pgd_acc = np.array(data)[:,1]
        mean = 100 * np.mean(list(nat_acc))
        err = 100 * np.std(list(nat_acc) / np.sqrt(len(nat_acc)))
        pgd_mean = 100 * np.mean(list(pgd_acc))
        pgd_err = 100 * np.std(list(pgd_acc) / np.sqrt(len(pgd_acc)))
        if latex:
            return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)+", "+"{:.1f} $\\pm$ {:.1f}".format(pgd_mean, pgd_err)
        else:
            return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)+ ", "+"{:.1f} +/-{:.1f}".format(pgd_mean, pgd_err)
    else:
        raise NotImplementedError

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex, task, attack):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"], task, attack) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        target_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*target_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            for j, target_env in enumerate(target_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, target_env",
                        (dataset, algorithm, target_env)
                    ).select("sweep_acc"))
                if trial_accs != []:
                    if selection_method == model_selection.OracleSelectionMethod:
                        if attack == 'None':
                            filename = os.path.join(args.input_dir, "best_model_path_clean.txt")
                        else:
                            filename = os.path.join(args.input_dir, "best_model_path_pgd.txt")
                    else:
                        raise NotImplementedError
                    with open(filename, 'w') as f:
                        for line in trial_accs:
                            f.write(line[-1]+'\n')
                    if attack == 'None':
                        _, _, table[i][j] = format_mean(np.array(np.array(trial_accs)[:,:1],dtype=np.float32), latex)
                    else:
                        _, _, table[i][j] = format_mean(np.array(np.array(trial_accs)[:,:2],dtype=np.float32), latex)

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=30, latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument('--task', type=str, default='domain_adaptation')
    parser.add_argument('--attack', type=str, default='None')
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = [
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex, args.task, args.attack)

    if args.latex:
        print("\\end{document}")
