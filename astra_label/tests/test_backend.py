"""Exercise the actual subprocess/file/cache boundary with a local fake executable."""
import json
from pathlib import Path
import tempfile
import unittest

from astra_label.backend import CodexBackend


class BackendTests(unittest.TestCase):
    def test_images_stdin_schema_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            executable = root / "fake_codex"
            executable.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
assert args[0] == "exec" and args[-1] == "-"
assert args[args.index("--model")+1] == "gpt-6-astra"
assert args[args.index("--sandbox")+1] == "read-only"
images = [args[i+1] for i, a in enumerate(args) if a == "--image"]
assert len(images) == 2 and all(pathlib.Path(p).is_file() for p in images)
schema = json.loads(pathlib.Path(args[args.index("--output-schema")+1]).read_text())
assert schema["type"] == "object"
out = pathlib.Path(args[args.index("--output-last-message")+1])
out.write_text(json.dumps({"seen_prompt": sys.stdin.read()}))
''')
            executable.chmod(0o755)
            images = [root / "first.png", root / "second.png"]
            for path in images:
                path.write_bytes(b"image content")
            backend = CodexBackend(root, executable=str(executable))
            schema = {"type": "object"}
            result = backend.ask("call", "first prompt", images, schema)
            self.assertEqual(result, {"seen_prompt": "first prompt"})
            # A cache hit must not need an executable. Changing prompt/images must invalidate it.
            executable.rename(root / "saved_executable")
            self.assertEqual(backend.ask("call", "first prompt", images, schema), result)
            with self.assertRaisesRegex(RuntimeError, "not found"):
                backend.ask("call", "changed prompt", images, schema)
            (root / "saved_executable").rename(executable)
            self.assertEqual(backend.ask("call", "changed prompt", images, schema), {"seen_prompt": "changed prompt"})
            images[0].write_bytes(b"changed pixels")
            executable.rename(root / "saved_executable")
            with self.assertRaisesRegex(RuntimeError, "not found"):
                backend.ask("call", "changed prompt", images, schema)

    def test_failure_does_not_consume_stale_response(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "calls" / "failure"
            folder.mkdir(parents=True)
            (folder / "response.json").write_text('{"stale": true}')
            backend = CodexBackend(root, executable="false")
            with self.assertRaisesRegex(RuntimeError, "Codex failed"):
                backend.ask("failure", "prompt", [], {"type": "object"})
            self.assertFalse((folder / "validated_transport.json").exists())
            self.assertFalse((folder / "response.json").exists())
