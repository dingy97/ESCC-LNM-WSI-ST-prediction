"""Fail if the release tree contains data, binary artifacts or obvious credentials."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".py", ".md", ".json", ".txt"}
SPECIAL = {".gitignore", ".gitattributes"}
SECRET = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def check(root):
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p in {".git", "__pycache__"} for p in relative.parts):
            continue
        if not path.is_file():
            continue
        if path.name not in SPECIAL and path.suffix not in ALLOWED:
            raise ValueError(f"Unexpected non-source file: {relative}")
        if path.stat().st_size > 512_000:
            raise ValueError(f"Unexpected large source file: {relative}")
        text = path.read_text(encoding="utf-8")
        if SECRET.search(text):
            raise ValueError(f"Credential-like content: {relative}")
        if re.search(r"[A-Z]:\\(?:Users|课题|空转原始数据|WSI医学图像)\\", text):
            raise ValueError(f"Unresolved private absolute path: {relative}")
        if path.suffix == ".py":
            ast.parse(text)
        files.append(relative.as_posix())
    return files


if __name__ == "__main__":
    print(f"Code-only check passed for {len(check(ROOT))} files")
