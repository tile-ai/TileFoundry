"""Spec command responses shared by installed core and target checks."""


def listed_documents(tf) -> str:
    done = tf("spec")
    assert done.returncode == 0, done.stderr
    return done.stdout


def check_rule(tf, topic, section, expected) -> None:
    done = tf("spec", topic, section)
    assert done.returncode == 0, done.stderr
    assert expected in done.stdout
