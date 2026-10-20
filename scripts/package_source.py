"""Build a source-only GitHub upload archive, without datasets or checkpoints."""

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
    ".gitattributes",
    "environment.project.yml",
    "requirements-cu128.txt",
)
PATTERNS = (
    ("src/dapi2marker", "*.py"),
    ("configs", "*.yaml"),
    ("tests", "*.py"),
    ("scripts", "*.py"),
    (".github", "*.yml"),
    ("splits", "*.json"),
)


def source_files(root=ROOT):
    paths = {root / name for name in TOP_LEVEL}
    paths.add(root / "docs" / "usage.md")
    paths.add(root / "docs" / "validation.md")
    paths.add(root / "docs" / "environment.md")
    for directory, pattern in PATTERNS:
        paths.update((root / directory).rglob(pattern))
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        if "__pycache__" not in path.parts:
            yield path


def main():
    output = ROOT / "artifacts" / "dapi2marker-source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(source_files())
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "dapi2marker/" + path.relative_to(ROOT).as_posix())
    temporary.replace(output)
    print(
        f"{output}\n{len(files)} source/config/test/manifest files; {output.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
