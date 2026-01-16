import os
import argparse
import pandas as pd
from know_subnet.utils import *

from subnet_test import get_config

def gather_results(
    nfs_dir,
    log_dir,
    project_name,
    is_random_baseline,
    expected_num_experiments,
    is_composition=False,
    composition_mode="seed",
    composition_kg=None,
    composition_seed=None,
    composition_pattern=None,
    max_ckpt_step=40000,
    delete_non_success_ckpts=False
):
    mydictlist = []
    bestdictlist = []

    work_dir = os.path.join(nfs_dir)
    project_path = os.path.join(work_dir, log_dir, project_name)
    log_path = os.path.join(project_path, "logs")
    
    for dirname in os.listdir(log_path):
        curr_dictlist = []
        exper_path = os.path.join(log_path, dirname)
        if not os.path.isdir(exper_path):
            continue
        
        print("-" * 40)
        print("Processing directory:", dirname)
        
        config = None
        config_path = os.path.join(exper_path, "config.json")
        if os.path.exists(config_path):
            config = get_config(exper_path)
        elif not is_composition:
            raise FileNotFoundError(f"Missing config.json in {exper_path}")
        else:
            config = {}
        res_list = []
        if is_random_baseline:
            res_files = [f for f in os.listdir(exper_path) if f.startswith("res-") and f.endswith("-random=True.csv")]
        else:
            res_files = [f for f in os.listdir(exper_path) if f.startswith("res-") and f.endswith("-random=False.csv")]
        
        print("Found result files:", res_files)
        
        assert len(res_files) == 1, f"Expected one result file, found {len(res_files)} in {exper_path}"
        
        res_filepath = os.path.join(exper_path, res_files[0])
        df = pd.read_csv(res_filepath)

        df = df[df["checkpoint_step"] <= max_ckpt_step]
        
        count_df = df[(df["controlkg-(inverse - full)-ppl"] <= 7.0)
            & (df["controllm-(inverse - full)-ppl"] <= 2.0)
            & (df["targetkg-(inverse - full)-ppl"] >= 40.0)]

        # 1) Get entries that satisfy the criteria
        criteria_df = df[(df["controlkg-(inverse - full)-ppl"] <= 5.0)
           & (df["controllm-(inverse - full)-ppl"] <= 1.0) 
           & (df["targetkg-(inverse - full)-ppl"] >= 35.0)]
        
        if criteria_df.empty:
            criteria_df = df[(df["controlkg-(inverse - full)-ppl"] <= 7.0)
                & (df["controllm-(inverse - full)-ppl"] <= 2.0)
                & (df["targetkg-(inverse - full)-ppl"] >= 40.0)]
            
            if criteria_df.empty:
                criteria_df = df[(df["controlkg-(inverse - full)-ppl"] <= 10.0)
                    & (df["controllm-(inverse - full)-ppl"] <= 3.0)
                    & (df["targetkg-(inverse - full)-ppl"] >= 45.0)]
                
                if criteria_df.empty:
                    criteria_df = df[(df["controlkg-(inverse - full)-ppl"] <= 15.0)
                        & (df["controllm-(inverse - full)-ppl"] <= 4.0)
                        & (df["targetkg-(inverse - full)-ppl"] >= 50.0)]
                    
                    if criteria_df.empty:
                        if max_ckpt_step in df["checkpoint_step"].values:
                            print(f"No entries met the criteria, defaulting to checkpoint step {max_ckpt_step}")
                            criteria_df = df[df["checkpoint_step"] == max_ckpt_step]
                        else:
                            print("No entries met the criteria, probably a test run, defaulting to the maximum checkpoint step available")
                            criteria_df = df[df["checkpoint_step"] == df["checkpoint_step"].max()]
        print(criteria_df.head())
        
        # 2) Iterate and save the necessary info in a new csv
        # if not criteria_df.empty:
        pattern_value = None
        if is_composition:
            if composition_pattern is not None:
                pattern_value = composition_pattern
            elif "composition-" in dirname:
                name_base = dirname.split("-date=")[0]
                pattern_value = name_base.split("-")[-1]

        for index, row in criteria_df.iterrows():
            if is_composition:
                if "eval_kg" in criteria_df.columns:
                    kg_value = row["eval_kg"]
                else:
                    kg_value = composition_kg or config.get("targetkg_name", "N/A")
                if "composition_seed" in criteria_df.columns:
                    seed_value = row["composition_seed"]
                else:
                    seed_value = composition_seed if composition_seed is not None else "combined"
            else:
                kg_value = config["targetkg_name"]
                seed_value = config["seed"]
            curr_dict = {
                "kg": kg_value,
                "seed": seed_value,
                "checkpoint step": row["checkpoint_step"],
                "sparsity": row["checkpoint_sparsity"],
                "score": row["targetkg-(inverse - full)-ppl"] - abs(row["controlkg-(inverse - full)-ppl"]) - abs(row["controllm-(inverse - full)-ppl"]),
                "targetkg-(inverse - full)-ppl": row["targetkg-(inverse - full)-ppl"],
                "controlkg-(inverse - full)-ppl": row["controlkg-(inverse - full)-ppl"],
                "controllm-(inverse - full)-ppl": row["controllm-(inverse - full)-ppl"],
                "full-targetkg-ppl": row["full-targetkg-ppl"],
                "full-controlkg-ppl": row["full-controlkg-ppl"],
                "full-controllm-ppl": row["full-controllm-ppl"],
                "ablation": config.get("subnet_config_file", "").split("/")[-1].replace(".json", "") if "ablation" in project_name else "N/A",
                "success_count": len(count_df)
            }
            if is_composition:
                curr_dict["pattern"] = pattern_value or "unknown"
            if not is_random_baseline and not is_composition:
                curr_dict["masked-targetkg-ppl"] = row["masked-targetkg-ppl"]
            curr_dictlist.append(curr_dict)
        
        mydictlist.extend(curr_dictlist)
        if is_composition:
            bestdictlist.extend(curr_dictlist)
        else:
            maxscoringcheckpoint = max(curr_dictlist, key=lambda x:x['score'])
            bestdictlist.append(maxscoringcheckpoint)
        
        # Optionally delete non-success checkpoints
        if delete_non_success_ckpts:
            if not criteria_df.empty:
                successful_steps = criteria_df["checkpoint_step"].tolist()
                for file in os.listdir(os.path.join(exper_path, "checkpoints")):
                    if file.startswith("ckpt-"): # and file.endswith(".pt"):
                        # ckpt-step=4-sparsity=1.00000
                        step = int(file.split("-")[1].split("=")[1])
                        if step not in successful_steps:
                            file_path = os.path.join(exper_path, "checkpoints", file)
                            os.remove(file_path)
                            print(f"Deleted non-success checkpoint: {file_path}")
            else:
                print(f"No successful checkpoints to retain in {exper_path}. Consider deleting all checkpoints if this was a test run.")
    
    print("Total number of experiments found:", len(bestdictlist))
    assert len(bestdictlist) == expected_num_experiments, f"Expected {expected_num_experiments} experiments, found {len(bestdictlist)}"
        
    all_path = os.path.join(project_path, f"success_all_ckpnts-random={is_random_baseline}.csv")
    best_path = os.path.join(project_path, f"success_best_ckpnts-random={is_random_baseline}.csv")
    
    save_dictlist_to_csv(mydictlist, all_path)
    save_dictlist_to_csv(bestdictlist, best_path)
    
    return best_path, project_path


def aggregate_ppl_results(
    best_path, 
    project_path, 
    is_random_baseline,
    convert_to_pct=True,
    group_by_key="kg"
):
    # Formatting for easy LaTeX conversion
    convert_col_title = {
        "kg": "\\textbf{Knowledge Graph}",
        "pattern": "\\textbf{Pattern}",
        "sparsity": "\\textbf{Sparsity} $\\uparrow$",
        "targetkg-(inverse - full)-ppl": "\\textbf{\\speckg{} PPL (Inverse - Full)} $\\uparrow$",
        "controlkg-(inverse - full)-ppl": "\\textbf{\\randkg{} PPL (Inverse - Full)} $\\downarrow$",
        "controllm-(inverse - full)-ppl": "\\textbf{\\lmodeling{} PPL (Inverse - Full)} $\\downarrow$",
    }
    convert_kg_col = {
        "building.n.01": "\\texttt{building.n.01}",
        "change.n.01": "\\texttt{change.n.01}",
        "communication.n.02": "\\texttt{communication.n.02}",
        "location.n.01": "\\texttt{location.n.01}",
        "magnitude.n.01": "\\texttt{magnitude.n.01}",
        "representation.n.02": "\\texttt{representation.n.02}",
        "statement.n.01": "\\texttt{statement.n.01}",
        "fruit": "\\texttt{fruit}",
        "sun": "\\texttt{sun}",
        "swim": "\\texttt{swim}",
    }

    success_df = pd.read_csv(best_path)
    print(success_df.head())
    print("Average TargetKG Delta PPL: ", success_df["targetkg-(inverse - full)-ppl"].mean())
    print("Average Sparsity: ",success_df["sparsity"].mean())
    success_stats_df = pd.DataFrame()
    
    metrics = [
        "sparsity",
        "targetkg-(inverse - full)-ppl",
        "controlkg-(inverse - full)-ppl",
        "controllm-(inverse - full)-ppl",
    ]

    for col in metrics:
        # Mean always per selected grouping key
        mean_col = success_df.groupby(group_by_key)[col].mean()

        # Min/Max: if ablation, get double-average then get min/max across KGs
        if group_by_key == "ablation":
            # average per (ablation, kg) first, then min/max across KGs for each ablation
            double_avg = success_df.groupby(["ablation", "kg"], as_index=False)[col].mean()
            min_col = double_avg.groupby("ablation")[col].min().reindex(mean_col.index)
            max_col = double_avg.groupby("ablation")[col].max().reindex(mean_col.index)
        else:
            min_col = success_df.groupby(group_by_key)[col].min()
            max_col = success_df.groupby(group_by_key)[col].max()
        
        if convert_to_pct and col == "sparsity":
            mean_col = mean_col * 100
            min_col = min_col * 100
            max_col = max_col * 100
        
        mean_col = mean_col.round(decimals=1)
        min_col = min_col.round(decimals=1)
        max_col = max_col.round(decimals=1)

        success_stats_df[col] = mean_col.astype('str') + ' [' + min_col.astype('str') + ", " + max_col.astype('str') +  ']'

    success_stats_df = success_stats_df.reset_index()
    success_stats_df["kg"] = success_stats_df["kg"].replace(to_replace=convert_kg_col)
    success_stats_df = success_stats_df.rename(columns=convert_col_title)

    print(success_stats_df.head())
    
    final_path = os.path.join(project_path, f"aggregated-random={is_random_baseline}.csv")
    success_stats_df.to_csv(final_path, index=False)


def aggregate_rankprob_results(best_path, project_path, is_random_baseline):
    pass


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="args for subnet_gather_results.py main func")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="folder to store training logs")
    parser.add_argument(
        "--nfs_dir",
        type=str,
        default="",
        help="nfs root dir")
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
        help="project directory that you want to gather results for")
    parser.add_argument(
        "--is_random_baseline",
        type=str2bool,
        default=False,
        help="whether doing random baseline calculation")
    parser.add_argument(
        "--expected_num_experiments",
        type=int,
        help="how many best checkpoints to expect in total for this project")
    parser.add_argument(
        "--is_composition",
        type=str2bool,
        default=False,
        help="whether gathering composition experiment results")
    parser.add_argument(
        "--composition_mode",
        type=str,
        default="seed",
        help="composition mode: seed or kg")
    parser.add_argument(
        "--composition_kg",
        type=str,
        default=None,
        help="target KG for seed composition gathering")
    parser.add_argument(
        "--composition_seed",
        type=int,
        default=None,
        help="seed for kg composition gathering")
    parser.add_argument(
        "--composition_pattern",
        type=str,
        default=None,
        help="pattern for composition gathering (intersection, union, floral)")
    parser.add_argument(
        "--group_by_key",
        type=str,
        default=None,
        help="grouping key for aggregation (kg, pattern, ablation)")
    parser.add_argument(
        "--delete_non_success_ckpts",
        type=str2bool,
        default=False,
        help="whether to delete non-successful checkpoints after gathering results")
    args, unknown = parser.parse_known_args()

    best_path, project_path = gather_results(
        nfs_dir=args.nfs_dir,
        log_dir=args.log_dir,
        project_name=args.project_name,
        is_random_baseline=args.is_random_baseline,
        expected_num_experiments=args.expected_num_experiments,
        is_composition=args.is_composition,
        composition_mode=args.composition_mode,
        composition_kg=args.composition_kg,
        composition_seed=args.composition_seed,
        composition_pattern=args.composition_pattern,
        max_ckpt_step=40000 if "scale" not in args.project_name else 50000,
        delete_non_success_ckpts=args.delete_non_success_ckpts
    )
    aggregate_ppl_results(
        best_path,
        project_path,
        is_random_baseline=args.is_random_baseline,
        group_by_key=(
            args.group_by_key
            if args.group_by_key is not None
            else ("ablation" if args.project_name == "ablation" else "kg")
        )
    )
    # aggregate_rankprob_results(best_path, project_path, is_random_baseline=args.is_random_baseline)


if __name__ == "__main__":
    main()
