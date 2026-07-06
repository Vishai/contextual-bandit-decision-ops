from contextual_bandit_decision_ops.smoke import project_name


def test_project_name():
    assert project_name() == "contextual-bandit-decision-ops"
