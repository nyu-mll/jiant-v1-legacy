import os
from shared_settings import (
    batch_size_to_accumulation,
    batch_size_limit_to_gpus,
    JIANT_PROJECT_PREFIX,
    JIANT_DATA_DIR,
    RANDOM_SEEDS,
    load_metadata,
    save_metadata,
    cpu_sbatch,
    basic_jiant_sbatch,
    num_main_trials,
    num_addi_trials,
    get_batch_size_limit,
)
from collect_trials import collect_trials


task_metadata = load_metadata()


def preprocess_tasks(input_module):
    outputs = [
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/spr1/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/spr2/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/dpr/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/dep_ewt/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/ontonotes/const/pos/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/ontonotes/const/nonterminal/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/ontonotes/srl/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/ontonotes/ner/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/ontonotes/coref/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="probing/retokenize_edge_data" ARGS="-t {input_module} {os.path.join(JIANT_DATA_DIR, "edges/semeval/*.json")}" sbatch {cpu_sbatch}',
        f'PROG="scripts/ccg/align_tags_to_bert" ARGS="-t {input_module} -d {os.path.join(JIANT_DATA_DIR, "ccg")}" sbatch {cpu_sbatch}',
    ]
    return outputs


def run_exp_init(input_module):
    outputs = []

    pretrain_tasks = ",".join(
        list(
            set(
                [
                    task_info["task_name"]
                    for task_info in task_metadata.values()
                    if ("I" in task_info["role"] or "M" in task_info["role"])
                ]
            )
        )
    )
    target_tasks = ",".join(
        list(
            set(
                [
                    task_info["task_name"]
                    for task_info in task_metadata.values()
                    if ("T" in task_info["role"] or "P" in task_info["role"])
                ]
            )
        )
    )

    run_name = "preprocess"

    for tasks, phase in zip([pretrain_tasks, target_tasks], ["pretrain", "target"]):
        exp_name = f"phase_{phase}_{input_module}"
        override = f'exp_name={exp_name}, run_name={run_name}, target_tasks=\\"{tasks}\\", input_module={input_module}'
        outputs.append(
            f'JIANT_OVERRIDES="{override}" sbatch --job-name={exp_name}.{run_name} {basic_jiant_sbatch}'
        )
    return outputs


def run_batch_size_check(input_module):
    outputs = []
    for batch_size in [32, 16, 8, 4, 2, 1]:
        task_names = list(set([task_info["task_name"] for task_info in task_metadata.values()]))
        for task_name in task_names:
            val_interval = task_metadata[task_name]["training_size"] // batch_size
            override = (
                f"exp_name=batch_size_{input_module}, run_name={task_name}_{batch_size}, "
                f"do_pretrain=1, pretrain_tasks={task_name}, "
                f"input_module={input_module}, batch_size={batch_size}, "
                f"max_epochs=1, val_interval={val_interval}, "
                f"delete_checkpoints_when_done=1"
            )
            outputs.append(f'JIANT_OVERRIDES="{override}" sbatch {basic_jiant_sbatch}')
    return outputs


def update_batch_size_check(input_module):
    exp_name = f"{input_module}_batch_size"
    task_batch_size_limit = {}
    for batch_size in [32, 16, 8, 4, 2, 1]:
        task_names = list(set([task_info["task_name"] for task_info in task_metadata.values()]))
        for task_name in task_names:
            run_name = f"{task_name}_{batch_size}"
            results_tsv = os.path.join(JIANT_PROJECT_PREFIX, exp_name, "results.tsv")
            if os.path.exists(results_tsv):
                with open(results_tsv, "r") as f:
                    results = dict([line.split("\t") for line in f.read().split("\n") if line])
                if run_name in results:
                    if (
                        task_name not in task_batch_size_limit
                        or batch_size > task_batch_size_limit[task_name]
                    ):
                        task_batch_size_limit[task_name] = batch_size
    for full_task_name, task_info in task_metadata.items():
        if task_info["task_name"] in task_batch_size_limit:
            batch_size_limit = task_batch_size_limit[task_info["task_name"]]
            task_info[f'{input_module.split("-")[0]}_batch_size_limit'] = batch_size_limit

    save_metadata(task_metadata)


def run_main_optuna_trials(input_module):
    outputs = []
    for full_task_name, task_info in task_metadata.items():
        if task_info["role"] == "M":
            continue
        print(input_module, full_task_name)
        df_grouped = collect_trials(full_task_name, input_module)[1]
        batch_size_limit = get_batch_size_limit(task_info, input_module)
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=False)
        training_size = task_info["training_size"]

        if df_grouped is None:
            previous_trials = 0
            print("previous trials not found")
        else:
            previous_trials = sum(df_grouped["count"])
            print(f"previous trials found: {previous_trials} / {num_main_trials}")

        remaining_trials = num_main_trials - previous_trials
        if remaining_trials <= 0:
            continue

        if training_size >= 100000 or remaining_trials > 6:
            parallel = 5
        elif training_size >= 20000 or remaining_trials > 4:
            parallel = 3
        else:
            parallel = 2

        for i in range(parallel):
            num_trials = (remaining_trials // parallel) + (i < (remaining_trials % parallel))
            if num_trials == 0:
                continue
            outputs.append(
                f'PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="'
                f"--study-name {full_task_name} --gpu-available {gpu_available} "
                f'--n-trials {num_trials} --input-module {input_module}" sbatch {sbatch}'
            )
    return outputs


def run_additional_optuna_trials(input_module):
    outputs = []
    for full_task_name, task_info in task_metadata.items():
        if task_info["role"] == "M":
            continue
        df_grouped = collect_trials(full_task_name, input_module)[1]
        if df_grouped is None or sum(df_grouped["count"]) < num_main_trials:
            print(f"{full_task_name} has not finished main optuna run.")
            continue
        batch_size_limit = get_batch_size_limit(task_info, input_module)
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=False)

        if df_grouped["count"][0] < num_addi_trials:
            for rank in range(3):
                num_trials = max(0, num_addi_trials - df_grouped["count"][rank])
                if num_trials == 0:
                    continue
                outputs.append(
                    f'PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="'
                    f"--study-name {full_task_name} --gpu-available {gpu_available} "
                    f'--max-epochs {int(df_grouped["max_epochs"][rank])} --lr {float(df_grouped["lr"][rank])} --batch-size {int(df_grouped["batch_size"][rank])} '
                    f'--n-trials {num_trials} --input-module {input_module}" sbatch {sbatch}'
                )
        else:
            task_metadata[full_task_name][f'{input_module.split("-")[0]}_hp'] = {
                "max_epochs": int(df_grouped["max_epochs"][0]),
                "lr": float(df_grouped["lr"][0]),
                "batch_size": int(df_grouped["batch_size"][0]),
            }

    save_metadata(task_metadata)
    return outputs


def run_pretrain(
    input_module,
    include_mlm=True,
    include_single_task=True,
    include_full_size=True,
    include_20k_size=True,
):
    exp_name = f"phase_pretrain_{input_module}"
    outputs = []
    checkpoints = {f"round{rid}": {} for rid, seed in enumerate(RANDOM_SEEDS)}

    for full_task_name, task_info in task_metadata.items():
        if "I" not in task_info["role"]:
            continue
        training_size = task_info["training_size"]
        if (not include_20k_size and "20k" in full_task_name) or (
            not include_full_size and training_size > 20000
        ):
            continue
        hp = task_info[f'{input_module.split("-")[0]}_hp']
        if hp == "":
            print(f"{full_task_name} {input_module} hp not available. skip task.")
            continue
        val_interval = max(training_size // hp["batch_size"], 5000)
        batch_size_limit = get_batch_size_limit(task_info, input_module)

        if include_single_task:
            gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=True)
            real_batch_size, accumulation_steps = batch_size_to_accumulation(
                batch_size_limit, hp["batch_size"], gpu_available
            )
            for rid, seed in enumerate(RANDOM_SEEDS):
                run_name = f"{full_task_name}_round{rid}"
                override = (
                    f"exp_name={exp_name}, run_name={run_name}, random_seed={seed}, load_model=1, "
                    f'do_pretrain=1, pretrain_tasks={task_info["task_name"]}, input_module={input_module}, '
                    f'max_epochs={hp["max_epochs"]}, lr={hp["lr"]}, '
                    f"batch_size={real_batch_size}, accumulation_steps={accumulation_steps}, "
                    f"val_interval={val_interval}"
                )
                outputs.append(
                    f'JIANT_OVERRIDES="{override}" sbatch --job-name={exp_name}.{run_name} {sbatch}'
                )
                checkpoints[f"round{rid}"][f"{full_task_name}"] = os.path.join(
                    JIANT_PROJECT_PREFIX, exp_name, run_name, "model_*.best.th"
                )

        if include_mlm:
            if input_module.split("-")[0] == "roberta":
                mlm_val_interval = val_interval * 2
                mlm_pretrain_tasks = f'\\"{task_info["task_name"]},wikipedia_corpus_mlm\\"'
                batch_size_limit = min(
                    batch_size_limit,
                    get_batch_size_limit(task_metadata["wikipedia_corpus_mlm"], input_module),
                )
            elif input_module.split("-")[0] == "albert":
                mlm_val_interval = val_interval * 3
                mlm_pretrain_tasks = (
                    f'\\"{task_info["task_name"]},wikipedia_corpus_mlm,wikipedia_corpus_sop\\"'
                )
                batch_size_limit = min(
                    batch_size_limit,
                    get_batch_size_limit(task_metadata["wikipedia_corpus_mlm"], input_module),
                    get_batch_size_limit(task_metadata["wikipedia_corpus_sop"], input_module),
                )
            gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=True)
            real_batch_size, accumulation_steps = batch_size_to_accumulation(
                batch_size_limit, hp["batch_size"], gpu_available
            )
            for rid, seed in enumerate(RANDOM_SEEDS):
                run_name = f"{full_task_name}_mtl_round{rid}"
                override = (
                    f"exp_name={exp_name}, run_name={run_name}, random_seed={seed}, load_model=1, "
                    f'do_pretrain=1, pretrain_tasks={mlm_pretrain_tasks}, target_tasks={task_info["task_name"]}, '
                    f'weighting_method=\\"examples_proportional_mixingK=16384\\", early_stopping={task_info["task_name"]}, '
                    f'input_module={input_module}, max_epochs={hp["max_epochs"]}, lr={hp["lr"]}, '
                    f"batch_size={real_batch_size}, accumulation_steps={accumulation_steps}, "
                    f"val_interval={mlm_val_interval}"
                )
                outputs.append(
                    f'JIANT_OVERRIDES="{override}" sbatch --job-name={exp_name}.{run_name} {sbatch}'
                )
                checkpoints[f"round{rid}"][f"{full_task_name}_mtl"] = os.path.join(
                    JIANT_PROJECT_PREFIX, exp_name, run_name, "model_*.best.th"
                )

    for rid, seed in enumerate(RANDOM_SEEDS):
        checkpoints[f"round{rid}"]["baseline"] = "none"

    return outputs, checkpoints


def run_target_train(
    input_module,
    pretrain_checkpoints,
    include_target=True,
    include_full_probing=True,
    include_5k_proibng=True,
):
    exp_name = f"phase_target_{input_module}"
    outputs = []

    for full_task_name, task_info in task_metadata.items():
        batch_size_limit = get_batch_size_limit(task_info, input_module)
        gpu_available, sbatch = batch_size_limit_to_gpus(batch_size_limit, jiant=True)
        hp = task_info[f'{input_module.split("-")[0]}_hp']
        real_batch_size, accumulation_steps = batch_size_to_accumulation(
            batch_size_limit, hp["batch_size"], gpu_available
        )
        training_size = task_info["training_size"]
        val_interval = max(training_size // hp["batch_size"], 5000)
        if (include_target and "T" in task_info["role"]) or (
            (include_full_probing and "P" in task_info["role"] and "5k" not in full_task_name)
            or (include_5k_proibng and "P" in task_info["role"] and training_size <= 5000)
        ):
            pass
        else:
            continue

        for pretrain_run_name in pretrain_checkpoints["round0"]:
            for rid, seed in enumerate(RANDOM_SEEDS):
                run_name = f"{full_task_name}_from_{pretrain_run_name}_round{rid}"
                override = (
                    f"exp_name={exp_name}, run_name={run_name}, random_seed={seed}, load_model=1, "
                    f'do_target_task_training=1, target_tasks={task_info["task_name"]}, input_module={input_module}, '
                    f'max_epochs={hp["max_epochs"]}, lr={hp["lr"]}, '
                    f"batch_size={real_batch_size}, accumulation_steps={accumulation_steps}, "
                    f"val_interval={val_interval}, "
                    f'load_target_train_checkpoint={pretrain_checkpoints[f"round{rid}"][pretrain_run_name]}'
                )
                outputs.append(
                    f'JIANT_OVERRIDES="{override}" sbatch --job-name={exp_name}.{run_name} {sbatch}'
                )

    return outputs


def write_script_file(script_name, outputs):
    script_name = os.path.join("scripts/taskmaster/submit_sbatch", script_name)
    with open(script_name, "w") as f:
        for line in outputs:
            f.write(line + "\n")


def sort_metadata():
    task_metadata = load_metadata()
    tm_tuple = list(task_metadata.items())

    def key_func(task_kv):
        if "training_size" in task_kv[1]:
            return task_kv[1]["training_size"]
        else:
            return -1

    tm_tuple.sort(reverse=True, key=key_func)
    task_metadata = dict(tm_tuple)
    save_metadata(task_metadata)
