import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_label.backend import CodexBackend
from astra_label.connection import API_KEY_ENV, Connection


class ConnectionTests(unittest.TestCase):
    def test_local_uses_saved_login_and_retains_proxy_without_changing_parent(self):
        with patch.dict(os.environ, {"CODEX_API_KEY": "unrelated", API_KEY_ENV: "custom",
                                     "HTTPS_PROXY": "http://proxy:8080"}):
            flags, env = Connection().process_settings()
            self.assertEqual(flags, [])
            self.assertNotIn("CODEX_API_KEY", env)
            self.assertNotIn(API_KEY_ENV, env)
            self.assertEqual(env["HTTPS_PROXY"], "http://proxy:8080")
            self.assertEqual(os.environ["CODEX_API_KEY"], "unrelated")

    def test_api_isolated_direct_connection_and_secret_never_in_command_or_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "private.json"
            path.write_text(json.dumps({"base_url": "https://example.test/v1", "api_key": "test-secret"}))
            with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy:8080", "http_proxy": "http://proxy:8080",
                                         "CODEX_API_KEY": "personal-override", API_KEY_ENV: ""}):
                connection = Connection("api", path)
                flags, env = connection.process_settings()
                self.assertEqual(env[API_KEY_ENV], "test-secret")
                self.assertNotIn("HTTPS_PROXY", env)
                self.assertNotIn("http_proxy", env)
                self.assertNotIn("CODEX_API_KEY", env)
                self.assertEqual(env["NO_PROXY"], "*")
                self.assertIn('model_providers.astra_label_api.requires_openai_auth=false', flags)
                self.assertIn('model_providers.astra_label_api.wire_api="responses"', flags)
                self.assertIn('--ignore-user-config', flags)
                self.assertNotIn("test-secret", json.dumps(flags) + json.dumps(connection.public()))
                self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy:8080")

    def test_api_environment_key_override_proxy_opt_in_and_no_login_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "private.json"
            path.write_text('{"api_key":"file-key"}')
            with patch.dict(os.environ, {API_KEY_ENV: "env-key", "HTTPS_PROXY": "http://proxy:8080"}):
                flags, env = Connection("api", path, api_use_proxy=True).process_settings()
                self.assertEqual(env[API_KEY_ENV], "env-key")
                self.assertEqual(env["HTTPS_PROXY"], "http://proxy:8080")
            path.write_text('{}')
            with patch.dict(os.environ, {API_KEY_ENV: ""}):
                with self.assertRaisesRegex(ValueError, "no login fallback"):
                    Connection("api", path).process_settings()

    def test_api_does_not_reuse_local_cache_and_redacts_provider_errors(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {API_KEY_ENV: ""}):
            root = Path(d)
            config = root / "credentials.json"
            config.write_text('{"base_url":"https://example.test/v1","api_key":"test-secret"}')
            executable = root / "codex"
            executable.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args=sys.argv[1:]
key=os.environ.get("ASTRA_LABEL_API_KEY", "local")
print(key)
print(key, file=sys.stderr)
path=pathlib.Path(args[args.index("--output-last-message")+1])
path.write_text(json.dumps({"mode": "local" if key == "local" else "api"}))
''')
            executable.chmod(0o755)
            schema = {"type": "object"}
            local = CodexBackend(root, executable=str(executable))
            self.assertEqual(local.ask("call", "prompt", [], schema), {"mode": "local"})
            api = CodexBackend(root, executable=str(executable), connection=Connection("api", config))
            self.assertEqual(api.ask("call", "prompt", [], schema), {"mode": "api"})
            for path in (root / "calls").rglob("*"):
                if path.is_file():
                    self.assertNotIn("test-secret", path.read_text(), str(path))
            self.assertIn("[REDACTED]", (root / "calls/call/stderr.log").read_text())
            # A different endpoint cannot reuse the first API endpoint's cached response.
            executable.unlink()
            other = CodexBackend(root, executable=str(executable),
                                 connection=Connection("api", config, "https://other.test/v1"))
            with self.assertRaisesRegex(RuntimeError, "not found"):
                other.ask("call", "prompt", [], schema)

    def test_reject_url_credentials(self):
        for url in ("http://example.test/v1", "https://user:key@example.test/v1", "https://example.test/v1?key=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Connection("api", api_base_url=url)
