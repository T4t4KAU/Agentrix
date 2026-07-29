from pathlib import Path

from coding_agent_tools import RepositoryTools


def test_public_test_expands_python_in_build_command(tmp_path: Path) -> None:
    agentrix = tmp_path / ".agentrix"
    agentrix.mkdir()
    public_test = agentrix / "public_test.py"
    public_test.write_text("raise SystemExit(0)\n", encoding="utf-8")
    task = {
        "allowed_paths": ["target.py"],
        "build": [
            {
                "cwd": ".",
                "argv": [
                    "{python}",
                    "-c",
                    "from pathlib import Path; Path('built').touch()",
                ],
            }
        ],
        "public_test_command": ["{python}", ".agentrix/public_test.py"],
        "timeout_seconds": 10,
    }

    event = RepositoryTools(tmp_path, task).public_test()

    assert '"returncode": 0' in event["content"]
    assert (tmp_path / "built").is_file()
