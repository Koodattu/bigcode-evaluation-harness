#!/usr/bin/env python3
import datetime
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

def round_numbers(obj):
    if isinstance(obj, dict):
        return {k: round_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_numbers(item) for item in obj]
    elif isinstance(obj, float):
        return round(obj, 2)
    else:
        return obj

def parse_json_from_output(output):
    """
    Extracts a valid JSON object from the subprocess output.
    This updated version scans through the output using JSONDecoder.raw_decode,
    which is useful when there is non-JSON content (such as for apps-introductory).
    """
    decoder = json.JSONDecoder()
    pos = 0
    valid_obj = None
    while pos < len(output):
        try:
            obj, pos_new = decoder.raw_decode(output, pos)
            valid_obj = obj  # save the most recent successfully decoded JSON object
            pos = pos_new
        except json.JSONDecodeError:
            pos += 1
    if valid_obj is not None:
        return valid_obj
    print("Error parsing JSON output: No valid JSON object found.")
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


def run_single_task_benchmark(model, task, common_args):
    """
    Runs the evaluation harness for a given model and a single task.
    Measures execution time and monitors VRAM usage independently.
    Returns a dictionary with elapsed time, max VRAM usage, and the parsed benchmark result.
    """
    model_name = model.split("/")[1]
    # Create directories if they don't exist
    os.makedirs("./results/references", exist_ok=True)
    os.makedirs("./results/generations", exist_ok=True)
    os.makedirs("./results/evaluations", exist_ok=True)
    common_args.extend(["--save_references_path", "./results/references/" + model_name + ".json"])
    common_args.extend(["--save_generations_path", "./results/generations/" + model_name + ".json"])
    common_args.extend(["--metric_output_path", "./results/evaluations/" + model_name + "_" + task + ".json"])

    command = ["accelerate", "launch", "main.py", "--model", model] + common_args + ["--tasks", task]
    print(f"\nRunning command for task '{task}':")
    print(" ".join(command))

    start_time = time.time()

    # Initialize NVML and get GPU 0 handle.
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # adjust index if needed
    except Exception as e:
        print("Error initializing NVML:", e)
        handle = None

    stop_event = threading.Event()
    max_vram_usage = [0]

    if handle is not None:
        monitor_thread = threading.Thread(target=monitor_vram, args=(handle, stop_event, 0.5, max_vram_usage))
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
    if handle is not None:
        monitor_thread.join()

    elapsed_time = time.time() - start_time
    vram_usage_mb = max_vram_usage[0] / (1024 * 1024) if max_vram_usage[0] else None

    benchmark_result = parse_json_from_output(output)

    # Clean the config from the benchmark result if it exists.
    if benchmark_result and "config" in benchmark_result:
        benchmark_result["config"] = clean_config(benchmark_result["config"])

    return {
        "total_elapsed_time_sec": elapsed_time,
        "max_vram_usage_mb": vram_usage_mb,
        "result": benchmark_result  # expected to include the task key and config
    }

def get_new_limit(task):
    """
    Returns a new limit for the given task.
    This is useful"
    """
    if task == "humaneval":
        return 164
    if task == "mbpp":
        return 1000
    if task == "mercury":
        return 1889
    return 1000

def main():
    # -------------------------------
    # Configuration Section
    # -------------------------------
    models = [
        "Qwen/Qwen2.5-Coder-0.5B",
        #"01-ai/Yi-Coder-1.5B"
    ]

    # List of tasks to run in a single call.
    tasks = [
        #"humaneval",
        #"mbpp",
        "mercury",
        #"codexglue_code_to_text-python-left",
        #"codexglue_code_to_text-go",
        #"codexglue_code_to_text-java",
        #"codexglue_code_to_text-javascript",
        #"codexglue_code_to_text-php",
        #"codexglue_code_to_text-python",
        #"codexglue_code_to_text-ruby",
        #"humanevalfixdocs-python",
        #"humanevalfixtests-python",
        #"humanevalsynthesize-python",
        #"",
        #"",
        #"",
        #"",
        #"",
        #"",
    ]

    # List of multi-run tasks
    tasks_multi = [
        "humanevalexplaindescribe-python",
        "humanevalexplainsynthesize-python",
    ]

    # HumanEval and MBPP can be ran with 512 tokens.
    # APPS requires 1024 tokens.

    # ---------- Generation Parameters ----------
    # Maximum token length for each evaluation instance (prompt + generation).
    # Increasing this value allows for longer outputs but raises memory usage and generation time.
    # For most tasks default 512 should be sufficient, but more complex tasks may require a higher value like 2048.
    MAX_LENGTH_GENERATION = 1024
    # Temperature controls the randomness in generation.
    # Lower values (near 0) yield more deterministic outputs, while higher values increase randomness.
    TEMPERATURE = 0.2
    # Number of generation samples per problem.
    # More samples can improve result diversity at the expense of increased computation.
    N_SAMPLES = 1
    # Batch size for generating outputs.
    # A larger batch size speeds up processing by generating in parallel but uses more memory.
    BATCH_SIZE = 1
    # Limit the number of problems to solve per task.
    # Useful for quick testing or reducing evaluation time.
    # Set to None to solve all problems.
    LIMIT = 1

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
    PRECISION = "fp16"
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
    # Run Benchmark for Each Model and Task
    # -------------------------------
    results = []
    for model in models:
        print(f"\n=== Benchmarking model: {model} ===")
        model_benchmark = {"model": model, "benchmark_result": {"tasks": {}}}
        common_config = None

        for task in tasks:
            print(f"\n--- Running task: {task} ---")
            result = run_single_task_benchmark(model, task, common_args)
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
                "total_tasks": new_limit * N_SAMPLES,
                "total_elapsed_time_sec": result["total_elapsed_time_sec"],
                "average_time_per_task_sec": result["total_elapsed_time_sec"] / (new_limit * N_SAMPLES),
                "max_vram_usage_mb": result["max_vram_usage_mb"],
                "result": task_result
            }

        # Attach the common configuration to the model-level benchmark.
        model_benchmark["benchmark_result"]["config"] = common_config
        results.append(model_benchmark)
        rounded_results = round_numbers(results)
        # Save intermediate results.
        timestamp = datetime.datetime.now().strftime("%d%m%Y-%H%M")
        filename = f"./results/benchmark_result_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(rounded_results, f, indent=2)

        print(f"\n=== Finished benchmarking model: {model} ===")

    print("\nAll benchmarks completed. Results saved to benchmark_results.json")

if __name__ == '__main__':
    main()