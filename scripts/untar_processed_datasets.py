#!/usr/bin/env python3
"""Extract dataset archives concurrently with native tar processes."""

import argparse
import concurrent.futures
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time


@dataclass(frozen=True)
class Archive:
    name: str
    parts: tuple[Path, ...]

    @property
    def size(self):
        return sum(part.stat().st_size for part in self.parts)


def discover_archives(directory):
    archives, split = {}, {}
    pattern = re.compile(r"^(.+\.tar)\.part(-?)([a-z]{2,})$")
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            name, separator, suffix = match.groups()
            split.setdefault(name, []).append((separator, suffix, path))
        elif path.name.endswith(".tar"):
            archives[path.name] = Archive(path.name, (path,))
    for name, entries in split.items():
        if name in archives:
            raise ValueError(f"Both {name} and split parts exist; remove the duplicate input first")
        entries.sort(key=lambda entry: entry[1])
        separator, first_suffix, _ = entries[0]
        for index, (part_separator, suffix, _) in enumerate(entries):
            number = 0
            for letter in suffix:
                number = number * 26 + ord(letter) - ord("a")
            if part_separator != separator or len(suffix) != len(first_suffix) or number != index:
                raise ValueError(f"Missing or mixed split parts for {name}; parts must start at {'a' * len(first_suffix)} and be consecutive")
        archives[name] = Archive(name, tuple(entry[2] for entry in entries))
    # Start the largest archives first to reduce the final long-running tail.
    return sorted(archives.values(), key=lambda archive: (-archive.size, archive.name))


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def extract_archive(archive, directory, logs, tar, cat, cancelled):
    if cancelled.is_set():
        return False, "cancelled"
    processes = []
    log_path = logs / f"{archive.name}.log"
    started = time.monotonic()
    try:
        with log_path.open("wb") as log:
            command = [tar, "--extract", "--ignore-zeros", "--file", "-", "--directory", str(directory)]
            if len(archive.parts) == 1 and archive.parts[0].name == archive.name:
                command[command.index("--file") + 1] = str(archive.parts[0])
                processes.append(subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log))
            else:
                producer = subprocess.Popen([cat, "--", *map(str, archive.parts)], stdout=subprocess.PIPE, stderr=log)
                processes.append(producer)
                try:
                    processes.append(subprocess.Popen(command, stdin=producer.stdout, stdout=subprocess.DEVNULL, stderr=log))
                finally:
                    producer.stdout.close()
            while any(process.poll() is None for process in processes):
                if cancelled.wait(0.2):
                    return False, "cancelled; input archives retained"
            codes = [process.wait() for process in processes]
            if any(codes):
                return False, f"exit codes {codes}; see {log_path}"
        return True, f"{time.monotonic() - started:.1f}s"
    except OSError as error:
        return False, str(error)
    finally:
        stop_processes(processes)


def flatten_datasets(directory):
    nested = directory / "processed_datasets"
    if not nested.exists():
        return
    if nested.is_symlink() or not nested.is_dir():
        raise ValueError(f"Refusing to move unexpected path: {nested}")
    children = list(nested.iterdir())
    for child in children:
        target = directory / child.name
        if target.exists() or target.is_symlink():
            raise ValueError(f"Cannot move {child}: destination {target} already exists; all inputs retained")
    for child in children:
        child.rename(directory / child.name)
    # Never recursively delete leftovers from a failed move.
    nested.rmdir()


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH") or os.environ.get("RBM_PROCESSED_DATASETS_PATH"), help="defaults to ROBOMETER_PROCESSED_DATASETS_PATH or RBM_PROCESSED_DATASETS_PATH")
    parser.add_argument("-j", "--jobs", type=positive_integer, default=95, help="parallel archive jobs (default: 95)")
    parser.add_argument("--keep-archives", action="store_true", help="retain input tar files and split parts after success")
    parser.add_argument("--dry-run", action="store_true", help="show the extraction plan without changing files")
    args = parser.parse_args(argv)
    if args.directory is None:
        parser.error("set ROBOMETER_PROCESSED_DATASETS_PATH (or RBM_PROCESSED_DATASETS_PATH), or pass --directory")
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"not a directory: {directory}")
    tar, cat = shutil.which("tar"), shutil.which("cat")
    if not tar or not cat:
        parser.error("tar and cat must be available on PATH")
    try:
        archives = discover_archives(directory)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    jobs = min(args.jobs, len(archives))
    print(f"Found {len(archives)} archives ({sum(len(a.parts) for a in archives)} input files), "
          f"{sum(a.size for a in archives) / 1024**4:.2f} TiB; "
          f"workers={jobs}, requested={args.jobs}", flush=True)
    if args.dry_run:
        for archive in archives:
            print(f"{archive.name}: {len(archive.parts)} part(s), {archive.size / 1024**3:.2f} GiB")
        return 0
    if not archives:
        print("No archives to extract.")
        return 0

    cancelled = threading.Event()
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda *_: cancelled.set())
    try:
        with (directory / ".untar.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another extraction is running in this directory.", flush=True)
                return 1
            logs = directory / ".untar-logs"
            logs.mkdir(exist_ok=True)
            failed = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(extract_archive, archive, directory, logs, tar, cat, cancelled): archive for archive in archives}
                for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    archive = futures[future]
                    ok, detail = future.result()
                    print(f"[{index}/{len(archives)}] {'OK' if ok else 'FAILED'} {archive.name}: {detail}", flush=True)
                    if not ok:
                        failed.append(archive.name)
            if failed or cancelled.is_set():
                print(f"Extraction incomplete: {len(failed)} failed/cancelled archive(s). All inputs retained.", flush=True)
                return 1
            flatten_datasets(directory)
            if not args.keep_archives:
                for archive in archives:
                    for part in archive.parts:
                        part.unlink()
            print("All archives extracted successfully. " + ("Inputs retained." if args.keep_archives else "Input archives removed."), flush=True)
            return 0
    except (OSError, ValueError) as error:
        print(f"Extraction failed: {error}", flush=True)
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
