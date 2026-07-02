from pathlib import Path

import pytest

import agent


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        "def retrieve():\n    return 4\n", encoding="utf-8"
    )
    return tmp_path


def test_dispatch_routes_grep(project: Path) -> None:
    result = agent.dispatch_tool("grep", {"pattern": "retrieve"}, project)

    assert "app.py:1" in result


def test_dispatch_routes_read_file(project: Path) -> None:
    result = agent.dispatch_tool("read_file", {"path": "app.py"}, project)

    assert result.startswith("1: def retrieve():")


def test_dispatch_routes_list_files(project: Path) -> None:
    result = agent.dispatch_tool("list_files", {}, project)

    assert "app.py" in result


def test_dispatch_reports_missing_required_arguments(project: Path) -> None:
    result = agent.dispatch_tool("grep", {}, project)

    assert "Tool error" in result


def test_dispatch_reports_sandbox_escapes_as_tool_errors(project: Path) -> None:
    result = agent.dispatch_tool("read_file", {"path": "../secrets.txt"}, project)

    assert "Tool error" in result


def test_dispatch_names_available_tools_for_unknown_names(project: Path) -> None:
    result = agent.dispatch_tool("delete_everything", {}, project)

    assert "Unknown tool" in result
    assert "grep" in result


def test_every_schema_is_a_complete_function_definition() -> None:
    names = set()

    for schema in agent.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"
        names.add(function["name"])

    assert names == {"list_files", "grep", "read_file"}
