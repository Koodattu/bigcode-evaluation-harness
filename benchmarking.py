#!/usr/bin/env python3
import datetime
import re
import subprocess
import time
import json
import threading
import sys
import os

try:
    import pynvml
except ImportError:
    print("Please install pynvml (pip install pynvml) to measure VRAM usage.")
    sys.exit(1)

def monitor_vram(handle, stop_event, interval, max_vram_usage):
    """
    Monitors VRAM usage on the given GPU handle until stop_event is set.
    The maximum observed usage (in bytes) is stored in max_vram_usage[0].
    """
    while not stop_event.is_set():
        try:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usage = meminfo.used  # in bytes
            if usage > max_vram_usage[0]:
                max_vram_usage[0] = usage
        except Exception as e:
            print("Error monitoring VRAM:", e)
        time.sleep(interval)

def monitor_vram_multi_gpu(handles, stop_event, interval, max_vram_usages):
    """
    Monitors VRAM usage on all provided GPU handles until stop_event is set.
    The maximum observed usage (in bytes) is stored in max_vram_usages[gpu_index].
    """
    while not stop_event.is_set():
        for gpu_index, handle in enumerate(handles):
            try:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                usage = meminfo.used  # in bytes
                if usage > max_vram_usages[gpu_index]:
                    max_vram_usages[gpu_index] = usage
            except Exception as e:
                print(f"Error monitoring VRAM on GPU {gpu_index}:", e)
        time.sleep(interval)

def round_numbers(obj):
    if isinstance(obj, dict):
        return {k: round_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_numbers(item) for item in obj]
    elif isinstance(obj, float):
        return round(obj, 2)
    else:
        return obj

def read_json_from_file(common_args):
    """
    Reads the JSON content from the given file path and returns the parsed JSON object.
    If the file cannot be read or parsed, an error message is printed and None is returned.
    """
    metric_output_index = common_args.index("--metric_output_path")
    metric_output_path = common_args[metric_output_index + 1]
    try:
        with open(metric_output_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file at {metric_output_path}: {e}")
        return None

def parse_json_from_output(output):
    """
    Attempts to extract the last JSON block printed by the harness.
    This function splits the output into lines, finds the first line starting with '{'
    (assuming that begins the JSON block), and then progressively trims lines from the bottom
    until the candidate can be parsed. Any standalone NaN values are replaced with null.
    """
    lines = output.splitlines()
    candidate_lines = []
    in_json = False
    for line in lines:
        if not in_json and line.lstrip().startswith("{"):
            in_json = True
        if in_json:
            candidate_lines.append(line)
    candidate = "\n".join(candidate_lines).strip()

    # Iteratively trim the candidate until json.loads succeeds.
    while candidate:
        try:
            # Replace standalone NaN with null.
            candidate_fixed = re.sub(r'\bNaN\b', 'null', candidate)
            return json.loads(candidate_fixed)
        except Exception as e:
            # If parsing fails, remove the last line and try again.
            candidate_lines = candidate_lines[:-1]
            candidate = "\n".join(candidate_lines).strip()
    print("No valid JSON found in output.")
    return None

def clean_config(config):
    """
    Removes keys from the config that are not needed in the output.
    """
    keys_to_remove = [
        "tasks", "model", "prefix", "do_sample", "top_k", "top_p", "eos", "seed",
        "modeltype", "peft_model", "revision", "use_auth_token", "trust_remote_code",
        "instruction_tokens", "left_padding", "limit_start", "save_every_k_tasks",
        "postprocess", "allow_code_execution", "generation_only", "load_generations_path",
        "load_data_path", "metric_output_path", "save_generations", "load_generations_intermediate_paths",
        "save_generations_path", "save_references", "save_references_path", "prompt",
        "max_memory_per_gpu", "check_references"
    ]
    for key in keys_to_remove:
        config.pop(key, None)
    return config

def update_arg(args, flag, new_value):
    """
    Update the value of a given flag in the args list.
    If the flag exists, update the following element.
    If it does not exist, append the flag and new_value.
    """
    if flag in args:
        idx = args.index(flag)
        # If there's a value following the flag, update it
        if idx + 1 < len(args):
            args[idx + 1] = new_value
        else:
            args.append(new_value)
    else:
        args.extend([flag, new_value])
    return args

def run_single_task_benchmark(model, task, common_args):
    """
    Runs the evaluation harness for a given model and a single task.
    Measures execution time and monitors VRAM usage independently.
    Returns a dictionary with elapsed time, max VRAM usage, and the parsed benchmark result.
    """
    model_name = model.split("/")[1]
    # Create directories if they don't exist
    common_args.extend(["--save_references_path", "./results/references/" + model_name + ".json"])
    common_args.extend(["--save_generations_path", "./results/generations/" + model_name + ".json"])
    common_args.extend(["--metric_output_path", "./results/evaluations/" + model_name + "_" + task + ".json"])

    # Only generation is needed for HumanEvalExplainDescribe tasks.
    if "humanevalexplaindescribe" in task:
        common_args.append("--generation_only")
    # Load the data path for HumanEvalExplainDescribe and HumanEvalExplainSynthesize tasks.
    if "humanevalexplainsynthesize" in task:
        prog_lang = task.split("-")[1]
        common_args.extend(["--load_data_path", "./results/generations/" + model_name + "_humanevalexplaindescribe-" + prog_lang + ".json"])

    # Used for HumanEvalPack with choices of: continue and instruct
    # HumanEvalFix and HumanEvalExplain require instruct
    # HumanEvalSynthesize requires continue
    if "humanevalfix" in task or "humanevalexplain" in task:
        common_args.extend(["--prompt", "instruct"])
    if "humanevalsynthesize" in task:
        common_args.extend(["--prompt", "continue"])
    # greedy decoding recommended for codexglue tasks
    if "codexglue" in task:
        common_args = update_arg(common_args, "--do_sample", "False")
        common_args = update_arg(common_args, "--n_samples", "1")
        common_args = update_arg(common_args, "--batch_size", "1")
    # mercury requires max length generation of 2048
    if "mercury" in task or "humanevalfix" in task:
        common_args = update_arg(common_args, "--max_length_generation", "1280")

    command = ["accelerate", "launch", "main.py", "--model", model] + common_args + ["--tasks", task]
    print(f"\nRunning command for task '{task}':")
    print(" ".join(command))

    start_time = time.time()

    # Initialize NVML and get handles for all GPUs.
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
    except Exception as e:
        print("Error initializing NVML:", e)
        handles = []

    stop_event = threading.Event()
    max_vram_usages = [0] * len(handles)

    if handles:
        monitor_thread = threading.Thread(target=monitor_vram_multi_gpu, args=(handles, stop_event, 0.5, max_vram_usages))
        monitor_thread.start()

    # Run the harness using Popen so that we can print output in real time.
    output = ""
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        output_lines = []
        for line in iter(proc.stdout.readline, ""):
            print(line, end="")  # print real-time output
            output_lines.append(line)
        proc.stdout.close()
        proc.wait()
        output = "".join(output_lines)
    except Exception as e:
        print("Error running benchmark for model", model, "on task", task, ":", e)

    stop_event.set()
    if handles:
        monitor_thread.join()

    elapsed_time = time.time() - start_time
    vram_usages_mb = [usage / (1024 * 1024) for usage in max_vram_usages]

    if "--generation_only" in common_args:
        benchmark_result = {"message": "humanevalexplaindescribe + generation_only = no output. humanevalexplainsynthesize will have output."}
    else:
        benchmark_result = read_json_from_file(common_args)

    # Clean the config from the benchmark result if it exists.
    if benchmark_result and "config" in benchmark_result:
        benchmark_result["config"] = clean_config(benchmark_result["config"])

    return {
        "total_elapsed_time_sec": elapsed_time,
        "max_vram_usage_mb_per_gpu": vram_usages_mb,
        "result": benchmark_result
    }

def get_new_limit(task):
    """
    Returns a new limit for calculating the total number of generations for a task when limit is None.
    """
    if "humaneval" in task:
        return 164
    if "mbpp" in task:
        return 500
    if "mercury" in task:
        return 256
    if "codexglue" in task:
        return 14918
    return 0

def main():
    # FOR SOME TASKS YOU NEED GO, JAVA, JAVASCRIPT, RUST
    # C++ = sudo apt install -y g++ clang libboost-all-dev
    # JAVASCRIPT = sudo apt install -y nodejs
    # RUST = sudo apt install -y rustc cargo
    # JAVA = sudo apt install -y default-jdk
    # GO = sudo apt install -y golang && GO111MODULE=off && go get github.com/stretchr/testify/assert

    os.makedirs("./results/references", exist_ok=True)
    os.makedirs("./results/generations", exist_ok=True)
    os.makedirs("./results/evaluations", exist_ok=True)
    
    # -------------------------------
    # Configuration Section
    # -------------------------------

    # safetensors, just the hf repo name
    st_models = [
        #"Qwen/Qwen2.5-Coder-0.5B",
        #"Qwen/Qwen2.5-Coder-7B",
        "01-ai/Yi-Coder-1.5B",
        #"01-ai/Yi-Coder-9B",
    ]

    # List of tasks to run in a single call.
    basic_tasks = [
        "humaneval",
        "mbpp",
    ]

    # Extra tasks
    extra_tasks = [
        "mercury",
    ]

    # CodeXGLUE Tasks
    tasks_codexglue = [
        "codexglue_code_to_text-python-left",
        "codexglue_code_to_text-python",
        "codexglue_code_to_text-go",
        "codexglue_code_to_text-java",
        "codexglue_code_to_text-javascript",
        "codexglue_code_to_text-php",
        "codexglue_code_to_text-ruby",
    ]

    # HumanEvalPack FixDocs Tasks
    tasks_fix_docs = [   
        "humanevalfixdocs-python",
        "humanevalfixdocs-cpp",
        "humanevalfixdocs-go",
        "humanevalfixdocs-java",
        "humanevalfixdocs-js",
        "humanevalfixdocs-rust",
    ]

    # HumanEvalPack FixTests Tasks
    tasks_fix_tests = [   
        "humanevalfixtests-python",
        "humanevalfixtests-cpp",
        "humanevalfixtests-go",
        "humanevalfixtests-java",
        "humanevalfixtests-js",
        "humanevalfixtests-rust",
    ]


    # HumanEvalPack Synthesize Tasks
    tasks_synth = [   
        "humanevalsynthesize-python",
        "humanevalsynthesize-cpp",
        "humanevalsynthesize-go",
        "humanevalsynthesize-java",
        "humanevalsynthesize-js",
        "humanevalsynthesize-rust",
    ]

    # HumanEvalPack Explain Tasks, Describe + Synthesize pairs
    tasks_explain = [
        "humanevalexplaindescribe-python",
        "humanevalexplainsynthesize-python",
        "humanevalexplaindescribe-cpp",
        "humanevalexplainsynthesize-cpp",
        "humanevalexplaindescribe-go",
        "humanevalexplainsynthesize-go",
        "humanevalexplaindescribe-java",
        "humanevalexplainsynthesize-java",
        "humanevalexplaindescribe-js",
        "humanevalexplainsynthesize-js",
        "humanevalexplaindescribe-rust",
        "humanevalexplainsynthesize-rust",
    ]

    all_tasks = basic_tasks + extra_tasks + tasks_codexglue + tasks_fix_docs + tasks_fix_tests + tasks_synth + tasks_explain

    # ---------- Model Configuration ----------
    # Possible values: causal, seq2seq (safetensors) or gguf
    MODEL_TYPE = "causal"

    # ---------- Generation Parameters ----------
    # Maximum token length for each evaluation instance (prompt + generation).
    # Increasing this value allows for longer outputs but raises memory usage and generation time.
    # For most tasks default 512 should be sufficient, but more complex tasks may require a higher value like 1024 or 2048.
    MAX_LENGTH_GENERATION = 512
    # Temperature controls the randomness in generation.
    # Lower values (near 0) yield more deterministic outputs, while higher values increase randomness.
    TEMPERATURE = 0.2
    # Number of generation samples per problem.
    # More samples can improve result diversity at the expense of increased computation.
    N_SAMPLES = 10
    # Batch size for generating outputs.
    # A larger batch size speeds up processing by generating in parallel but uses more memory.
    BATCH_SIZE = 10
    # Limit the number of problems to solve per task.
    # Useful for quick testing or reducing evaluation time.
    # Set to None to solve all problems.
    LIMIT = 3

    # ---------- Execution and Saving Options ----------
    # Allow the execution of generated code.
    # Use with caution: this enables running external/untrusted Python code.
    ALLOW_CODE_EXECUTION = True
    # Save the post-processed generated code to disk.
    # This helps with later inspection or debugging of the generated solutions.
    SAVE_GENERATIONS = True
    # Save the reference (gold standard) solutions/tests.
    SAVE_REFERENCES = True

    # ----------- Precision Settings -----------
    # Precision for model computations.
    # Lower precision can speed up computations and lower memory usage.
    # Options include "fp32", "fp16", or "bf16".
    PRECISION = "bf16"
    # Load the model in 8-bit mode to reduce memory footprint.
    LOAD_IN_8BIT = False
    # Load the model in 4-bit mode for further memory savings.
    LOAD_IN_4BIT = False

    # ---------- Additional Generation & Model Settings ----------
    # Toggle sampling mode.
    # If True, generation will use sampling; otherwise, it might use greedy decoding.
    DO_SAMPLE = True
    # Trust remote code when loading models.
    # Some models from the Hugging Face Hub require executing custom code.
    TRUST_REMOTE_CODE = True
    # Use Hugging Face authentication token.
    # Set to True when accessing private models.
    USE_AUTH_TOKEN = False

    # -------------------------------
    # Build Command-Line Arguments
    # -------------------------------
    # Helper function to add an argument if its value is not None.
    def add_arg(flag, value):
        if value is not None:
            common_args.extend([flag, str(value)])

    common_args = []
    # Generation and model configuration parameters.
    add_arg("--max_length_generation", MAX_LENGTH_GENERATION)
    add_arg("--temperature", TEMPERATURE)
    add_arg("--n_samples", N_SAMPLES)
    add_arg("--batch_size", BATCH_SIZE)
    add_arg("--limit", LIMIT)
    add_arg("--precision", PRECISION)
    add_arg("--do_sample", DO_SAMPLE)
    add_arg("--modeltype", MODEL_TYPE)

    # Boolean flags: only added if set to True.
    if ALLOW_CODE_EXECUTION:
        common_args.append("--allow_code_execution")
    if SAVE_GENERATIONS:
        common_args.append("--save_generations")
    if SAVE_REFERENCES:
        common_args.append("--save_references")
    if TRUST_REMOTE_CODE:
        common_args.append("--trust_remote_code")
    if LOAD_IN_8BIT:
        common_args.append("--load_in_8bit")
    if LOAD_IN_4BIT:
        common_args.append("--load_in_4bit")
    if USE_AUTH_TOKEN:
        common_args.append("--use_auth_token")

    # -------------------------------
    # Prepare to Save Benchmark Results
    # -------------------------------
    results = []
    # Create the benchmark file name once using the current timestamp.
    timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M")
    filename = f"./results/benchmark_result_{timestamp}.json"
    def update_json_file(results, filename):
        with open(filename, "w") as f:
            json.dump(round_numbers(results), f, indent=2)

    # -------------------------------
    # Run Benchmark for Each Model and Task
    # -------------------------------
    for model in st_models:
        print(f"\n=== Benchmarking model: {model} ===")
        model_benchmark = {"model": model, "benchmark_result": {"tasks": {}}}
        results.append(model_benchmark)
        common_config = None
        for task in all_tasks:
            print(f"\n--- Running task: {task} ---")
            result = run_single_task_benchmark(model, task, common_args.copy())
            # The benchmark result from the run is expected to contain the task key and a config.
            task_result = None
            if result["result"]:
                # Extract the result for the task if it exists;
                # otherwise, use the entire benchmark result.
                task_result = result["result"].get(task, result["result"])
                # Save the common config if not already saved.
                if common_config is None and "config" in result["result"]:
                    common_config = result["result"]["config"]

            new_limit = LIMIT if LIMIT is not None else get_new_limit(task)
            model_benchmark["benchmark_result"]["tasks"][task] = {
                "total_generations": new_limit * N_SAMPLES,
                "total_elapsed_time_sec": result["total_elapsed_time_sec"],
                "average_time_per_generation_sec": result["total_elapsed_time_sec"] / (new_limit * N_SAMPLES),
                "max_vram_usage_mb_per_gpu": result["max_vram_usage_mb_per_gpu"],
                "max_vram_usage_mb_total": sum(result["max_vram_usage_mb_per_gpu"]),
                "result": task_result
            }
            update_json_file(results, filename)

        # Attach the common configuration to the model-level benchmark.
        model_benchmark["benchmark_result"]["config"] = common_config
        update_json_file(results, filename)

        print(f"\n=== Finished benchmarking model: {model} ===")

    print("\nAll benchmarks completed. Results saved to benchmark_results.json")

if __name__ == '__main__':
    main()