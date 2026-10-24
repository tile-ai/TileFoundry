"""Target command assertions shared by installed core and vendor tests."""


def check_target_list(tf, values, imports) -> None:
    done = tf("target", "list")
    assert done.returncode == 0, done.stderr
    for expression, identity in values:
        assert expression in done.stdout
        assert f"identity: {identity}" in done.stdout
    for statement in imports:
        assert statement in done.stdout
    assert done.stderr == ""


def check_unknown_identity(tf, identities) -> None:
    done = tf("target", "show", "vendor.missing")
    assert done.returncode == 1
    assert done.stdout == ""
    for identity in identities:
        assert identity in done.stderr
