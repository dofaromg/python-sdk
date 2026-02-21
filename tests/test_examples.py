"""Tests for example servers"""
# TODO(Marcelo): The `examples` directory needs to be importable as a package.
# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false

import sys
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from pytest_examples import CodeExample, EvalExample, find_examples

from mcp import Client
from mcp.types import CallToolResult, TextContent, TextResourceContents


@pytest.mark.anyio
async def test_simple_echo():
    """Test the simple echo server"""
    from examples.mcpserver.simple_echo import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("echo", {"text": "hello"})
        assert result == snapshot(
            CallToolResult(content=[TextContent(text="hello")], structured_content={"result": "hello"})
        )


@pytest.mark.anyio
async def test_complex_inputs():
    """Test the complex inputs server"""
    from examples.mcpserver.complex_inputs import mcp

    async with Client(mcp) as client:
        tank = {"shrimp": [{"name": "bob"}, {"name": "alice"}]}
        result = await client.call_tool("name_shrimp", {"tank": tank, "extra_names": ["charlie"]})
        assert result == snapshot(
            CallToolResult(
                content=[
                    TextContent(text="bob"),
                    TextContent(text="alice"),
                    TextContent(text="charlie"),
                ],
                structured_content={"result": ["bob", "alice", "charlie"]},
            )
        )


@pytest.mark.anyio
async def test_direct_call_tool_result_return():
    """Test the CallToolResult echo server"""
    from examples.mcpserver.direct_call_tool_result_return import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("echo", {"text": "hello"})
        assert result == snapshot(
            CallToolResult(
                meta={"some": "metadata"},  # type: ignore[reportUnknownMemberType]
                content=[TextContent(text="hello")],
                structured_content={"text": "hello"},
            )
        )


@pytest.mark.anyio
async def test_desktop(monkeypatch: pytest.MonkeyPatch):
    """Test the desktop server"""
    # Mock desktop directory listing
    mock_files = [Path("/fake/path/file1.txt"), Path("/fake/path/file2.txt")]
    monkeypatch.setattr(Path, "iterdir", lambda self: mock_files)  # type: ignore[reportUnknownArgumentType]
    monkeypatch.setattr(Path, "home", lambda: Path("/fake/home"))

    from examples.mcpserver.desktop import mcp

    async with Client(mcp) as client:
        # Test the sum function
        result = await client.call_tool("sum", {"a": 1, "b": 2})
        assert result == snapshot(CallToolResult(content=[TextContent(text="3")], structured_content={"result": 3}))

        # Test the desktop resource
        result = await client.read_resource("dir://desktop")
        assert len(result.contents) == 1
        content = result.contents[0]
        assert isinstance(content, TextResourceContents)
        assert isinstance(content.text, str)
        if sys.platform == "win32":  # pragma: no cover
            file_1 = "/fake/path/file1.txt".replace("/", "\\\\")  # might be a bug
            file_2 = "/fake/path/file2.txt".replace("/", "\\\\")  # might be a bug
            assert file_1 in content.text
            assert file_2 in content.text
            # might be a bug, but the test is passing
        else:  # pragma: lax no cover
            assert "/fake/path/file1.txt" in content.text
            assert "/fake/path/file2.txt" in content.text


@pytest.mark.anyio
async def test_curve_comparison_smooth():
    """Test the smooth_curve tool"""
    from examples.mcpserver.curve_comparison import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("smooth_curve", {"values": [1.0, 2.0, 3.0, 4.0, 5.0], "window": 3})
        assert result == snapshot(
            CallToolResult(
                content=[TextContent(text="2.0"), TextContent(text="3.0"), TextContent(text="4.0")],
                structured_content={"result": [2.0, 3.0, 4.0]},
            )
        )


@pytest.mark.anyio
async def test_curve_comparison_smooth_invalid_window():
    """Test smooth_curve raises error for invalid window"""
    from examples.mcpserver.curve_comparison import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("smooth_curve", {"values": [1.0, 2.0, 3.0], "window": 0})
        assert result.is_error is True

        result = await client.call_tool("smooth_curve", {"values": [1.0, 2.0], "window": 5})
        assert result.is_error is True


@pytest.mark.anyio
async def test_curve_comparison_compare():
    """Test the compare_curves tool"""
    from examples.mcpserver.curve_comparison import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("compare_curves", {"curve1": [1.0, 2.0, 3.0], "curve2": [2.0, 3.0, 4.0]})
        assert result == snapshot(
            CallToolResult(
                content=[TextContent(text='{\n  "pearson_correlation": 1.0,\n  "dtw_distance": 2.0\n}')],
                structured_content={"pearson_correlation": 1.0, "dtw_distance": 2.0},
            )
        )


@pytest.mark.anyio
async def test_curve_comparison_compare_truncates():
    """Test that compare_curves truncates to the shorter curve"""
    from examples.mcpserver.curve_comparison import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("compare_curves", {"curve1": [1.0, 2.0, 3.0, 4.0], "curve2": [1.0, 2.0, 3.0]})
        assert result.is_error is False
        assert result.structured_content == snapshot({"pearson_correlation": 1.0, "dtw_distance": 0.0})


@pytest.mark.anyio
async def test_curve_comparison_compare_too_short():
    """Test compare_curves raises error when curves are too short"""
    from examples.mcpserver.curve_comparison import mcp

    async with Client(mcp) as client:
        result = await client.call_tool("compare_curves", {"curve1": [1.0], "curve2": [2.0, 3.0]})
        assert result.is_error is True


# TODO(v2): Change back to README.md when v2 is released
@pytest.mark.parametrize("example", find_examples("README.v2.md"), ids=str)
def test_docs_examples(example: CodeExample, eval_example: EvalExample):
    ruff_ignore: list[str] = ["F841", "I001", "F821"]  # F821: undefined names (snippets lack imports)

    # Use project's actual line length of 120
    eval_example.set_config(ruff_ignore=ruff_ignore, target_version="py310", line_length=120)

    # Use Ruff for both formatting and linting (skip Black)
    if eval_example.update_examples:  # pragma: no cover
        eval_example.format_ruff(example)
    else:
        eval_example.lint_ruff(example)
