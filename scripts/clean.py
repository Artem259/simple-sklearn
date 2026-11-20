import os
import shutil
import sys
from pathlib import Path

ABSOLUTE_TARGETS = frozenset(
    {
        ".cache",
        ".mypy_cache_notebooks",
        "act-artifacts",
        "build",
        "dist",
        "site",
        ".coverage",
        "src/simple_sklearn.egg-info",
    }
)
RELATIVE_TARGETS = frozenset(
    {
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

SKIP_DIRS = frozenset({".git", "venv", ".venv", ".idea"})
BYTECODE_SUFFIXES = (".pyc", ".pyo", ".pyd", "$py.class")


def _remove_dir(path: Path) -> bool:
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    return True


def _remove_file(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _clean_dirs(root: Path, cur_path: Path, dirs: list[str], removed: list[str], failed: list[str]) -> int:
    pycache_count = 0
    for d in list(dirs):
        dir_path = cur_path / d
        dir_rootpath = dir_path.relative_to(root).as_posix()

        if dir_rootpath in ABSOLUTE_TARGETS or d in RELATIVE_TARGETS:
            dirs.remove(d)
            (removed if _remove_dir(dir_path) else failed).append(f"{dir_rootpath}/")
        elif d == "__pycache__":
            dirs.remove(d)
            if _remove_dir(dir_path):
                pycache_count += 1
            else:
                failed.append(f"{dir_rootpath}/")
    return pycache_count


def _clean_files(root: Path, cur_path: Path, files: list[str], removed: list[str], failed: list[str]) -> int:
    bytecode_count = 0
    for f in files:
        file_path = cur_path / f
        file_rootpath = file_path.relative_to(root).as_posix()

        if file_rootpath in ABSOLUTE_TARGETS or f in RELATIVE_TARGETS:
            (removed if _remove_file(file_path) else failed).append(file_rootpath)
        elif f.endswith(BYTECODE_SUFFIXES):
            if _remove_file(file_path):
                bytecode_count += 1
            else:
                failed.append(file_rootpath)
    return bytecode_count


def _clean(root: Path) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    failed: list[str] = []
    pycache_count = 0
    bytecode_count = 0

    for cur_dir, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        cur_path = Path(cur_dir)

        pycache_count += _clean_dirs(root, cur_path, dirs, removed, failed)
        bytecode_count += _clean_files(root, cur_path, files, removed, failed)

    if pycache_count:
        removed.append(f"{pycache_count} __pycache__ dir(s)")
    if bytecode_count:
        removed.append(f"{bytecode_count} bytecode file(s)")
    return removed, failed


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    removed, failed = _clean(root)
    if removed:
        print("Removed:", ", ".join(removed))
    elif not failed:
        print("Workspace is already clean.")
    if failed:
        print("Failed to remove:", ", ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
