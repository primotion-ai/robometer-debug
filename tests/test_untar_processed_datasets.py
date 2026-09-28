import fcntl
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/untar_processed_datasets.sh"
REAL_TAR = shutil.which("tar")
REAL_CAT = shutil.which("cat")


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, name, members):
        path = self.root / name
        with tarfile.open(path, "w") as archive:
            for filename, content in members.items():
                member = tarfile.TarInfo(filename)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return path

    def split(self, archive, separator):
        data = archive.read_bytes()
        parts = []
        for i, offset in enumerate(range(0, len(data), 4093)):
            part = archive.with_name(archive.name + ".part" + separator + "a" + chr(ord("a") + i))
            part.write_bytes(data[offset:offset + 4093])
            parts.append(part)
        archive.unlink()
        return parts

    def run_script(self, *args, env=None):
        return subprocess.run(["bash", str(SCRIPT), "--directory", str(self.root), *args],
                              env=env, capture_output=True, text=True, timeout=30)

    def wrapper(self, name, body):
        directory = self.root / "bin"
        directory.mkdir(exist_ok=True)
        executable = directory / name
        executable.write_text(f"#!{sys.executable}\n" + body)
        executable.chmod(0o755)
        return dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"])

    def test_mixed_split_formats_spaces_and_flattening(self):
        content = os.urandom(13000)
        regular = self.archive("regular archive.tar", {"dataset one/data.bin": content})
        parts_a = self.split(self.archive("split a.tar", {"processed_datasets/dataset two/data.bin": content}), "")
        parts_b = self.split(self.archive("split b.tar", {"dataset three/data.bin": content}), "-")
        result = self.run_script("--jobs", "95", "--keep-archives")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("workers=3, requested=95", result.stdout)
        for name in ("dataset one", "dataset two", "dataset three"):
            self.assertEqual((self.root / name / "data.bin").read_bytes(), content)
        self.assertTrue(all(p.exists() for p in [regular, *parts_a, *parts_b]))
        self.assertFalse((self.root / "processed_datasets").exists())

    def test_default_cleanup_only_after_success(self):
        source = self.archive("a.tar", {"dataset/data": b"complete"})
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.root / "dataset/data").read_bytes(), b"complete")
        self.assertFalse(source.exists())

    def test_corrupt_archive_keeps_all_inputs_and_returns_failure(self):
        good = self.archive("good.tar", {"good/data": b"good"})
        bad = self.root / "bad.tar"
        bad.write_bytes(b"not a tar archive")
        result = self.run_script("--jobs", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(good.exists())
        self.assertTrue(bad.exists())
        self.assertTrue((self.root / ".untar-logs/bad.tar.log").read_text())

    def test_missing_split_part_is_rejected_before_extraction(self):
        parts = self.split(self.archive("a.tar", {"dataset/data": b"data"}), "-")
        parts[1].unlink()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing or mixed split parts", result.stderr)
        self.assertFalse((self.root / "dataset").exists())
        self.assertTrue(parts[0].exists())

    def test_cat_failure_is_not_hidden_by_successful_tar(self):
        parts = self.split(self.archive("a.tar", {"dataset/data": b"data"}), "-")
        env = self.wrapper("cat", f"import subprocess, sys\nsubprocess.run([{REAL_CAT!r}, *sys.argv[1:]], check=True)\nraise SystemExit(7)\n")
        result = self.run_script(env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(all(p.exists() for p in parts))
        self.assertIn("exit codes [7, 0]", result.stdout)

    def test_native_tar_processes_really_overlap(self):
        self.archive("a.tar", {"a/data": b"a"})
        self.archive("b.tar", {"b/data": b"b"})
        barrier = self.root / "barrier"
        barrier.mkdir()
        env = self.wrapper("tar", f"""import os, sys, time
from pathlib import Path
barrier = Path({str(barrier)!r})
(barrier / str(os.getpid())).touch()
deadline = time.monotonic() + 10
while len(list(barrier.iterdir())) < 2:
    if time.monotonic() > deadline:
        raise SystemExit(91)
    time.sleep(0.05)
os.execv({REAL_TAR!r}, [{REAL_TAR!r}, *sys.argv[1:]])
""")
        result = self.run_script("--jobs", "2", env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.root / "a/data").read_bytes(), b"a")
        self.assertEqual((self.root / "b/data").read_bytes(), b"b")

    def test_dry_run_changes_nothing(self):
        self.archive("a.tar", {"a/data": b"a"})
        before = sorted(self.root.iterdir())
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(self.root.iterdir()), before)

    def test_flatten_conflict_preserves_both_directories_and_archive(self):
        source = self.archive("a.tar", {"processed_datasets/existing/new": b"new"})
        (self.root / "existing").mkdir()
        (self.root / "existing/old").write_bytes(b"old")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "existing/old").read_bytes(), b"old")
        self.assertEqual((self.root / "processed_datasets/existing/new").read_bytes(), b"new")
        self.assertTrue(source.exists())

    def test_directory_lock_blocks_a_second_run(self):
        source = self.archive("a.tar", {"a/data": b"a"})
        with (self.root / ".untar.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(source.exists())
        self.assertFalse((self.root / "a").exists())


if __name__ == "__main__":
    unittest.main()
