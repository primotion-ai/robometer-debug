"""CPU-only reproduction of GC scanning RandomSampler's large index list.

The values are sequential to omit randperm cost; the list/int object layout is
the same as randperm(...).tolist(). No model, dataset, CUDA, or W&B is loaded.
"""
import argparse
import gc
import json
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-length", type=int, default=1_041_377)
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 10])
    args = parser.parse_args()
    if args.base_length < 1 or any(n < 1 for n in args.repeats):
        parser.error("base-length and repeats must be positive")
    enabled = gc.isenabled()
    try:
        for repeats in args.repeats:
            gc.collect()
            gc.disable()
            size = args.base_length * repeats
            print(json.dumps({"phase": "allocate", "index_count": size,
                              "estimated_list_and_int_gib": size * 36 / 2**30}), flush=True)
            started = time.perf_counter()
            indices = list(range(size))
            build_s = time.perf_counter() - started
            for generation in (0, 1, 2):
                started, cpu_started = time.perf_counter(), time.thread_time()
                gc.collect(generation)
                print(json.dumps({"index_count": size, "generation": generation,
                                  "build_s": build_s, "gc_wall_s": time.perf_counter() - started,
                                  "gc_cpu_s": time.thread_time() - cpu_started,
                                  "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}), flush=True)
            del indices
    finally:
        (gc.enable if enabled else gc.disable)()


if __name__ == "__main__":
    main()
