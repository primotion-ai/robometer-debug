"""Official Codex CLI transport, structured responses, and exact-request caching."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .connection import Connection


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class CodexBackend:
    def __init__(self, output, model="gpt-6-astra", reasoning="high", timeout=900, executable="codex", connection=None):
        self.output = Path(output)
        self.model = model
        self.reasoning = reasoning
        self.timeout = timeout
        self.executable = executable
        self.connection = connection or Connection()

    def ask(self, name, prompt, images, schema):
        folder = self.output / "calls" / name
        folder.mkdir(parents=True, exist_ok=True)
        request = {"prompt": prompt, "schema": schema, "model": self.model, "reasoning": self.reasoning,
                   "images": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in images]}
        # Keep legacy local cache keys valid; never reuse local results for a custom provider.
        if self.connection.mode == "api":
            request["connection"] = self.connection.public()
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cache = folder / "validated_transport.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved["request_sha256"] == digest:
                print(f"Reuse {name}", flush=True)
                return saved["response"]
        write_json(folder / "request.json", request)
        (folder / "prompt.txt").write_text(prompt)
        schema_path = (folder / "schema.json").resolve()
        write_json(schema_path, schema)
        response_path = (folder / "response.json").resolve()
        # A fresh attempt must not accidentally consume a previous response.
        response_path.unlink(missing_ok=True)
        workspace = (folder / "workspace").resolve()
        workspace.mkdir(exist_ok=True)
        if not shutil.which(self.executable):
            raise RuntimeError(f"Codex CLI not found: {self.executable}; install Codex and run codex login")
        flags, env = self.connection.process_settings()
        command = [self.executable, "exec", *flags, "--model", self.model,
                   "-c", f'model_reasoning_effort="{self.reasoning}"',
                   "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                   "--cd", str(workspace), "--output-schema", str(schema_path),
                   "--output-last-message", str(response_path), "--json", "--color", "never"]
        for path in images:
            command.extend(["--image", str(path.resolve())])
        command.append("-")
        write_json(folder / "command.json", command)
        print(f"Call {name}: {self.model}, {len(images)} images, {self.connection.public()} "
              f"(timeout {self.timeout}s)", flush=True)
        try:
            result = subprocess.run(command, input=prompt, text=True, capture_output=True, env=env,
                                    timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            (folder / "events.jsonl").write_text(self.connection.redact(exc.stdout))
            (folder / "stderr.log").write_text(self.connection.redact(exc.stderr))
            if response_path.exists():
                response_path.write_text(self.connection.redact(response_path.read_text()))
            raise RuntimeError(f"Codex timed out; inspect {folder}. Rerun to reuse completed calls.") from exc
        (folder / "events.jsonl").write_text(self.connection.redact(result.stdout))
        (folder / "stderr.log").write_text(self.connection.redact(result.stderr))
        if response_path.exists():
            response_path.write_text(self.connection.redact(response_path.read_text()))
        if result.returncode or not response_path.exists():
            detail = ""
            for line in self.connection.redact(result.stdout).splitlines():
                try:
                    event = json.loads(line)
                    if event.get("type") == "turn.failed":
                        detail = str(event.get("error", {}).get("message", ""))[:600]
                except (ValueError, AttributeError):
                    continue
            raise RuntimeError(f"Codex failed (exit {result.returncode}): {detail}; "
                               f"inspect {folder}/stderr.log and events.jsonl")
        response = json.loads(response_path.read_text())
        write_json(cache, {"request_sha256": digest, "response": response})
        return response
