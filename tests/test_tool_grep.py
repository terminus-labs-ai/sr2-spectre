"""Tests for GrepTool."""
import pytest

from sr2_spectre.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Schema / class-attribute contract
# ---------------------------------------------------------------------------

def test_grep_class_attributes() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    assert isinstance(GrepTool.name, str) and GrepTool.name
    assert isinstance(GrepTool.description, str) and GrepTool.description
    assert isinstance(GrepTool.input_schema, dict)


def test_grep_name_is_grep() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    assert GrepTool.name == "grep"


def test_grep_input_schema_properties() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    props = GrepTool.input_schema.get("properties", {})
    assert "pattern" in props
    assert "path" in props
    assert "glob" in props
    assert "regex" in props


def test_grep_input_schema_required() -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    required = GrepTool.input_schema.get("required", [])
    # pattern is the only required input
    assert "pattern" in required
    assert "path" not in required
    assert "glob" not in required
    assert "regex" not in required
    assert required == ["pattern"]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_grep_registers_via_class_path() -> None:
    reg = ToolRegistry()
    reg.register_from_class_path("sr2_spectre.tools.builtins.grep.GrepTool")
    assert "grep" in reg


# ---------------------------------------------------------------------------
# Literal single-file match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_literal_single_file_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    target.write_text(
        "alpha line\nneedle here\ngamma line\n",
        encoding="utf-8",
    )

    tool = GrepTool()
    result = await tool(pattern="needle", path=str(target), regex=False)

    assert isinstance(result, str)
    # The matched line is on line 2 (1-based) with text "needle here".
    assert "2:needle here" in result
    # The file's name appears in the output.
    assert "file.txt" in result


# ---------------------------------------------------------------------------
# Regex match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_regex_match(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    # "foo" and "fao" both match r"f.o"; "literal" does not.
    target.write_text("foo\nfao\nliteral\n", encoding="utf-8")

    tool = GrepTool()
    # regex=True is the default.
    result = await tool(pattern=r"f.o", path=str(target))

    assert isinstance(result, str)
    # Regex dot matched real characters, not requiring a literal "f.o".
    assert "1:foo" in result
    assert "2:fao" in result
    # A literal search of "f.o" would match neither line, but regex does.
    assert "f.o" not in result


# ---------------------------------------------------------------------------
# Literal mode treats metacharacters literally
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_literal_mode_treats_metachars_literally(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    # Line 1 is the literal "a.b"; line 2 is "axb" which a regex "a.b" WOULD match.
    target.write_text("a.b\naxb\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="a.b", path=str(target), regex=False)

    assert isinstance(result, str)
    # Only the literal "a.b" line matches.
    assert "1:a.b" in result
    # "axb" must NOT be returned in literal mode.
    assert "axb" not in result


# ---------------------------------------------------------------------------
# Recursive directory search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_recursive_directory_search(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    nested_file = nested / "file.txt"
    nested_file.write_text("nothing\nfindme please\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="findme", path=str(tmp_path), regex=False)

    assert isinstance(result, str)
    assert "file.txt" in result
    assert "2:findme please" in result


# ---------------------------------------------------------------------------
# Multiple matches across multiple files
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_multiple_files_both_appear(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.write_text("token is here\n", encoding="utf-8")
    file_two.write_text("also token\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="token", path=str(tmp_path), regex=False)

    assert isinstance(result, str)
    assert "one.txt" in result
    assert "two.txt" in result


# ---------------------------------------------------------------------------
# Multiple matches within one file (different lines)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_multiple_matches_within_one_file(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    target.write_text(
        "hit one\nnothing\nhit two\n",
        encoding="utf-8",
    )

    tool = GrepTool()
    result = await tool(pattern="hit", path=str(target), regex=False)

    assert isinstance(result, str)
    # Match on line 1 and line 3.
    assert "1:hit one" in result
    assert "3:hit two" in result


# ---------------------------------------------------------------------------
# glob scope
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_glob_scopes_files(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    py_file = tmp_path / "a.py"
    txt_file = tmp_path / "b.txt"
    py_file.write_text("the pattern lives here\n", encoding="utf-8")
    txt_file.write_text("the pattern also here\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(
        pattern="pattern", path=str(tmp_path), regex=False, glob="*.py"
    )

    assert isinstance(result, str)
    assert "a.py" in result
    assert "b.txt" not in result


# ---------------------------------------------------------------------------
# No-match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_no_match_returns_string_not_raise(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tool = GrepTool()
    # Should not raise on zero matches.
    result = await tool(pattern="absent_pattern", path=str(target), regex=False)

    assert isinstance(result, str)
    # Non-error string that signals the absence of matches. Accept common
    # phrasings ("no match", "no matches", "nothing matched") without pinning
    # exact wording. The absent pattern must NOT appear as a reported match.
    lowered = result.lower()
    assert "no match" in lowered or "nothing match" in lowered
    assert ":absent_pattern" not in result


# ---------------------------------------------------------------------------
# Single-file path (path points directly at a file)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_single_file_path(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "solo.txt"
    target.write_text("first\nmatch_target\nthird\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="match_target", path=str(target), regex=False)

    assert isinstance(result, str)
    # Path passed in points directly at the file; that path appears in output.
    assert str(target) in result
    assert "2:match_target" in result


# ---------------------------------------------------------------------------
# Binary file is skipped, not fatal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_binary_file_skipped(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    binary = tmp_path / "binary.bin"
    binary.write_bytes(b"\xff\xfe\x00\x01")

    text_file = tmp_path / "text.txt"
    text_file.write_text("normal\nhas pattern\n", encoding="utf-8")

    tool = GrepTool()
    # Must not raise on the undecodable binary file.
    result = await tool(pattern="pattern", path=str(tmp_path), regex=False)

    assert isinstance(result, str)
    assert "text.txt" in result
    assert "2:has pattern" in result


# ---------------------------------------------------------------------------
# Line numbers are 1-based
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grep_line_numbers_are_one_based(tmp_path) -> None:
    from sr2_spectre.tools.builtins.grep import GrepTool

    target = tmp_path / "file.txt"
    # The match is on the 3rd line.
    target.write_text("one\ntwo\nNEEDLE\nfour\n", encoding="utf-8")

    tool = GrepTool()
    result = await tool(pattern="NEEDLE", path=str(target), regex=False)

    assert isinstance(result, str)
    # 1-based: the match must be reported on line 3, formatted ":3:".
    assert ":3:NEEDLE" in result
    # And it must NOT be reported on a 0-based or off-by-one ":2:".
    assert ":2:NEEDLE" not in result
