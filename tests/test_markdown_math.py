"""Prevent regression of GitHub-rendered mathematical Markdown."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FILES = ("README.md", "docs/mathematics.md")
FENCE = chr(96) * 3


def _math_blocks(content: str) -> list[str]:
    """Extract fenced math while also checking all Markdown fences close."""
    blocks = []
    mode = None
    body = []
    for line_number, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(FENCE):
            if mode is None:
                mode = stripped[len(FENCE):].strip()
                body = []
            else:
                assert stripped == FENCE, f"Unexpected nested fence at line {line_number}"
                if mode == "math":
                    blocks.append("\n".join(body))
                mode = None
        elif mode is not None:
            body.append(line)
    assert mode is None, "Unclosed Markdown code/math fence"
    return blocks


def _assert_balanced_braces(equation: str) -> None:
    depth = 0
    for index, char in enumerate(equation):
        escaped = index > 0 and equation[index - 1] == "\\"
        if char == "{" and not escaped:
            depth += 1
        elif char == "}" and not escaped:
            depth -= 1
            assert depth >= 0, "Unmatched closing brace in a math expression"
    assert depth == 0, "Unmatched opening brace in a math expression"
    assert re.findall(r"\\begin\{([^}]+)\}", equation) == re.findall(
        r"\\end\{([^}]+)\}", equation
    ), "Unmatched LaTeX environment"


def test_markdown_math_is_github_compatible():
    for relative in FILES:
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert r"\operatorname" not in content, f"{relative}: unsupported operatorname macro"
        assert not re.search(r"(?m)^\s*\$\s*$", content), f"{relative}: lone $ math delimiter"
        assert not re.search(r"(?m)^\s*\$\$\s*$", content), f"{relative}: use fenced math blocks"
        assert not re.search(r"(?<!\\)mathrm\{", content), f"{relative}: escaped LaTeX backslash missing"
        blocks = _math_blocks(content)
        assert blocks, f"{relative}: no GitHub math fences found"
        for equation in blocks:
            assert equation.strip(), f"{relative}: empty math block"
            _assert_balanced_braces(equation)
