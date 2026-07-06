import json

from contextual_bandit_decision_ops import __version__
from contextual_bandit_decision_ops.metadata_cli import main, project_metadata


def test_project_metadata_matches_package_version() -> None:
    metadata = project_metadata()

    assert metadata["name"] == "contextual-bandit-decision-ops"
    assert metadata["version"] == __version__
    assert metadata["service_mode"] == "local-staging-only"
    assert metadata["cpu_only"] is True


def test_version_command(capsys) -> None:
    assert main(["--version"]) == 0

    assert capsys.readouterr().out.strip() == __version__


def test_metadata_command(capsys) -> None:
    assert main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == __version__
