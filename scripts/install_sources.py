"""Install source into a private analysis workspace without copying research data."""
import argparse
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def render_source(text, paths):
    for token, path in paths.items():
        # Tokens replace the contents of original raw or normal Python literals.
        # Forward slashes are valid on Windows and avoid accidental escape sequences.
        text = text.replace(token, Path(path).as_posix())
    return text


def install(project_root, paths, *, overwrite=False, dry_run=False):
    source_root = REPOSITORY / "analysis"
    changes = []
    for source in sorted(source_root.rglob("*.py")):
        target = project_root / source.relative_to(source_root)
        target.resolve().relative_to(project_root.resolve())
        text = render_source(source.read_text(encoding="utf-8"), paths)
        remaining = [part.split("__", 1)[0] for part in text.split("__ESCC_")[1:]]
        if remaining:
            raise ValueError(f"Missing path settings for {source.relative_to(source_root)}")
        if target.exists() and target.read_text(encoding="utf-8") == text:
            continue
        if target.exists() and not overwrite:
            raise FileExistsError(f"Different local code exists at {target}. Use --overwrite-code only after retaining your source revision.")
        changes.append((source, target, text))
    # Validate the entire write set before performing any writes.
    for _, target, text in changes:
        if dry_run:
            print(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".release-tmp")
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.replace(target)
    return len(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True, help="Private JSON following config/paths.example.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-code", action="store_true")
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text(encoding="utf-8"))
    required = {
        "__ESCC_PROJECT_ROOT__", "__ESCC_LEGACY_ROOT__", "__ESCC_SURGICAL_WSI_ROOT__",
        "__ESCC_ESD_WSI_ROOT__", "__ESCC_TCGA_WSI_ROOT__", "__ESCC_USER_HOME__",
        "__ESCC_COSMX_HE_WSI__", "__ESCC_RAW_COSMX_ROOT__",
    }
    if set(paths) != required:
        raise ValueError(f"Provide exactly these path tokens: {sorted(required)}")
    root = Path(paths["__ESCC_PROJECT_ROOT__"]).resolve()
    n = install(root, paths, overwrite=args.overwrite_code, dry_run=args.dry_run)
    print(f"{'Planned' if args.dry_run else 'Installed'} {n} source files")


if __name__ == "__main__":
    main()
