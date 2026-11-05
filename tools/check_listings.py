"""
Check that every function and class printed in the book matches the repository.

    python tools/check_listings.py path/to/manuscript/

For each ```python block in the manuscript's Markdown files, every top-level
function and class is looked up by name in the repository and compared by its
syntax tree (formatting and comments are ignored; code and docstrings are not).
Exits with status 1 if anything printed is missing or different.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEARCH = ["algo_builder", "adapters", "chapters"]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def definitions(tree: ast.AST) -> dict[str, ast.AST]:
    return {n.name: n for n in getattr(tree, "body", [])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def repo_index() -> dict[str, list[tuple[Path, str]]]:
    index: dict[str, list[tuple[Path, str]]] = {}
    for folder in SEARCH:
        for path in (REPO / folder).rglob("*.py"):
            for name, node in definitions(ast.parse(path.read_text(encoding="utf-8"))).items():
                index.setdefault(name, []).append((path, ast.dump(node)))
    return index


def main(manuscript: Path) -> int:
    index, problems, checked = repo_index(), [], 0
    for md in sorted(manuscript.glob("*.md")):
        for n, code in enumerate(BLOCK.findall(md.read_text(encoding="utf-8")), 1):
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                problems.append(f"{md.name} block {n}: does not parse ({e.msg})")
                continue
            for name, node in definitions(tree).items():
                checked += 1
                found = index.get(name)
                if not found:
                    problems.append(f"{md.name} block {n}: `{name}` not in the repository")
                elif ast.dump(node) not in {d for _, d in found}:
                    where = ", ".join(str(p.relative_to(REPO)) for p, _ in found)
                    problems.append(f"{md.name} block {n}: `{name}` differs from {where}")
    for p in problems:
        print("MISMATCH", p)
    print(f"{checked} printed definitions checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "manuscript")))
