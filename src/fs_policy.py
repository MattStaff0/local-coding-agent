"""One shared ignore/deny policy for every filesystem-facing feature.

Two functions with deliberately different teeth:

- denied(path)  — the security boundary. Secrets and binary-class artifacts
  never reach the model, even when the user explicitly attaches them.
- ignored(root, path) — search hygiene. Live list/grep skip these, but an
  explicit @path attachment overrides: typing the path IS the intent.

Kept dependency-free on purpose: .gitignore support is a minimal fnmatch
subset (per-line patterns, `dir/` anchors, comments). `!` negation is
unsupported and skipped — documented in the README.
"""
import fnmatch
from pathlib import Path

# Directories no tool should ever descend into, .gitignore or not.
SKIP_DIRS = {".git", "__pycache__", "chroma_db", ".venv", "node_modules"}

# Filename patterns that are never readable: secrets and credential material.
_DENY_NAMES = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "*credentials*",
    "*secret*",
]

# Binary-class artifacts: weights, databases, datasets, images, archives.
# Useless as model context and often huge; datasets may also carry PII.
_DENY_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".gguf", ".onnx", ".ckpt",
    ".sqlite", ".sqlite3", ".db", ".parquet", ".feather",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".whl",
    ".so", ".dylib", ".dll", ".bin",
}


def denied(relative: Path) -> str | None:
    """Why this path may never be read, or None when it is allowed."""
    if ".git" in relative.parts:
        return "matches deny rule '.git/'"

    name = relative.name.lower()
    for pattern in _DENY_NAMES:
        if fnmatch.fnmatch(name, pattern):
            return f"matches deny rule '{pattern}'"

    suffix = relative.suffix.lower()
    if suffix in _DENY_SUFFIXES:
        return f"matches deny rule '*{suffix}'"

    return None


def _gitignore_patterns(root: Path) -> list[str]:
    """Patterns from the root .gitignore; blank, comment, and '!' lines skipped."""
    try:
        text = (root / ".gitignore").read_text(encoding="utf-8")
    except OSError:
        return []

    patterns = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line)
    return patterns


def ignored(root: Path, relative: Path) -> bool:
    """Whether live search should skip this root-relative path."""
    if any(part in SKIP_DIRS for part in relative.parts):
        return True

    posix = relative.as_posix()
    for pattern in _gitignore_patterns(root):
        if pattern.endswith("/"):
            directory = pattern.strip("/")
            if "/" in directory:
                # Nested dir pattern ("generated/cache/") anchors to the root.
                if posix == directory or posix.startswith(directory + "/"):
                    return True
            elif any(fnmatch.fnmatch(part, directory) for part in relative.parts):
                # Bare dir pattern ("build/") matches at any depth, like git.
                return True
        else:
            name_pattern = pattern.lstrip("/")
            if fnmatch.fnmatch(relative.name, name_pattern) or fnmatch.fnmatch(
                posix, name_pattern
            ):
                return True

    return False
