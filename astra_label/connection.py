"""Per-process authentication/provider settings; never modify Codex's saved login."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_API_CONFIG = Path(__file__).resolve().parent / "api.local.json"
DEFAULT_BASE_URL = "https://ckffai.com/v1"
API_KEY_ENV = "ASTRA_LABEL_API_KEY"
OVERRIDE_ENV = (
    "CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN", "OPENAI_BASE_URL",
    "OPENAI_FEDERATION_RULE_ID", "OPENAI_IDENTITY_TOKEN_FILE", "OPENAI_WORKLOAD_IDENTITY_CONTEXT",
)


def add_connection_arguments(parser):
    parser.add_argument("--auth-mode", choices=("local", "api"), help="local Codex login (default), or custom API")
    parser.add_argument("--api-config", type=Path, help="private JSON containing base_url and api_key")
    parser.add_argument("--api-base-url", help="override the custom API base URL")
    parser.add_argument("--api-use-proxy", action="store_true", default=None,
                        help="inherit proxy settings in API mode (default: direct connection)")


def connection_overrides(args):
    return {name: str(value) if isinstance(value, Path) else value
            for name in ("auth_mode", "api_config", "api_base_url", "api_use_proxy")
            if (value := getattr(args, name, None)) is not None}


class Connection:
    def __init__(self, auth_mode="local", api_config=None, api_base_url=None, api_use_proxy=False):
        if auth_mode not in ("local", "api"):
            raise ValueError("auth_mode must be local or api")
        if type(api_use_proxy) is not bool:
            raise ValueError("api_use_proxy must be boolean")
        self.mode = auth_mode
        self.use_proxy = api_use_proxy
        self.base_url = None
        self._key = None
        if auth_mode == "api":
            path = Path(api_config).expanduser() if api_config else DEFAULT_API_CONFIG
            settings = {}
            if path.exists():
                try:
                    settings = json.loads(path.read_text())
                except (ValueError, OSError) as exc:
                    raise ValueError(f"Cannot read API credentials config: {path}") from exc
                if not isinstance(settings, dict) or set(settings) - {"base_url", "api_key"}:
                    raise ValueError("API config must contain only base_url and api_key")
            elif api_config:
                raise ValueError(f"API credentials config not found: {path}")
            self.base_url = api_base_url or settings.get("base_url", DEFAULT_BASE_URL)
            if not isinstance(self.base_url, str):
                raise ValueError("API base URL must be a string")
            url = urlsplit(self.base_url)
            if (url.scheme != "https" or not url.hostname or url.username or url.password
                    or url.query or url.fragment or any(c.isspace() for c in self.base_url)):
                raise ValueError("API base URL must be HTTPS without embedded credentials, query or fragment")
            self.base_url = self.base_url.rstrip("/")
            self._key = os.environ.get(API_KEY_ENV) or settings.get("api_key")
            if self._key is not None and (not isinstance(self._key, str) or not self._key.strip()):
                raise ValueError("API key must be a nonempty string")

    @classmethod
    def from_config(cls, config):
        return cls(**{key: config[key] for key in ("auth_mode", "api_config", "api_base_url", "api_use_proxy")
                      if key in config})

    def public(self):
        return {"auth_mode": self.mode, "base_url": self.base_url,
                "proxy": "inherit" if self.mode == "local" or self.use_proxy else "direct"}

    def process_settings(self):
        env = os.environ.copy()
        for key in (*OVERRIDE_ENV, API_KEY_ENV):
            env.pop(key, None)
        if self.mode == "local":
            return [], env
        if not self._key:
            raise ValueError(f"API mode requires {API_KEY_ENV} or api_key in {DEFAULT_API_CONFIG}; no login fallback")
        env[API_KEY_ENV] = self._key
        if not self.use_proxy:
            for name in list(env):
                if name.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
                    env.pop(name)
            env.update({"NO_PROXY": "*", "no_proxy": "*"})
        settings = {
            "model_provider": "astra_label_api",
            "model_providers.astra_label_api.name": "Astra annotation API",
            "model_providers.astra_label_api.base_url": self.base_url,
            "model_providers.astra_label_api.env_key": API_KEY_ENV,
            "model_providers.astra_label_api.requires_openai_auth": False,
            "model_providers.astra_label_api.wire_api": "responses",
            "model_providers.astra_label_api.supports_websockets": False,
            "cli_auth_credentials_store": "ephemeral",
        }
        flags = ["--ignore-user-config"]
        for key, value in settings.items():
            flags.extend(["-c", f"{key}={json.dumps(value)}"])
        return flags, env

    def redact(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return (text or "").replace(self._key, "[REDACTED]") if self._key else (text or "")
