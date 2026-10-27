"""One reading of the `--docker-arg` values, used by everything that must understand them: the masking of environment values in the reports and the refusal of
the flags the suite sets itself. Two readers would disagree (`-u0`, `-te K=V`, `-v/x:/y` are all valid docker spellings), and the disagreement is a leak or a false PASS.

It follows pflag, which parses `docker run`: a short option takes its value attached (`-u0`, `-u=0`) or from the next argument (`-u 0`); boolean short options
can be clustered (`-it`, `-tu0`); a long option takes `=value` or the next argument.
"""
from typing import NamedTuple

BOOLEAN_SHORT = frozenset("diqtP")  # docker run's short flags without a value
VALUE_SHORT = frozenset("acehlmpuvw")  # and with one


class Option(NamedTuple):
    index: int  # of the argument that holds the option
    flag: str  # "-u", "--user" ("-?x" for a short flag the parser does not know)
    head: str  # the text of that argument before the value: "-tu", "-e=", "--env="
    value: str | None  # attached value; None when the value is the next argument or there is none
    value_index: int | None  # index of the argument holding the value when it is the next one


def options(args: list[str]):
    """Every option of the docker arguments, in order (everything after `--` is not ours to read)."""
    i = 0
    while i < len(args):
        arg, here = args[i], i
        i += 1
        if arg == "--":
            return
        if arg.startswith("--"):
            name, eq, value = arg.partition("=")
            if not eq and name == "--env" and i < len(args):  # the one long option whose separate value matters here (secrets); any other value is not dash-prefixed
                yield Option(here, name, arg, None, i)
                i += 1
            else:
                yield Option(here, name, name + eq, value if eq else None, None)
        elif arg.startswith("-") and len(arg) > 1:
            for pos, ch in enumerate(arg[1:], 1):
                if ch in BOOLEAN_SHORT:
                    yield Option(here, "-" + ch, arg[:pos + 1], None, None)
                elif ch in VALUE_SHORT:
                    rest = arg[pos + 1:]
                    if rest:  # `-u0`, `-u=0`, `-tev=x`
                        eq = rest.startswith("=")
                        yield Option(here, "-" + ch, arg[:pos + 1] + ("=" if eq else ""), rest[1:] if eq else rest, None)
                    elif i < len(args):  # `-u 0`: the value is the next argument, which is not an option
                        yield Option(here, "-" + ch, arg, None, i)
                        i += 1
                    else:
                        yield Option(here, "-" + ch, arg, None, None)
                    break
                else:
                    yield Option(here, "-?" + ch, arg, None, None)
                    break


def env_options(args: list[str]):
    """(index of the argument that holds the value, text before the value in it, "NAME[=VALUE]") for every environment variable passed to the container."""
    for o in options(args):
        if o.flag not in ("-e", "--env"):
            continue
        if o.value is not None:
            yield o.index, o.head, o.value
        elif o.value_index is not None:
            yield o.value_index, "", args[o.value_index]


def unparseable(args: list[str]) -> str | None:
    """The first cluster of short flags this reader cannot be sure about: it could hide an environment value or a flag we refuse. Returned as its flag letters up to
    the unknown one and an ellipsis, never the rest of the argument (which may be a secret: `-xeTOKEN=abc`)."""
    for o in options(args):
        if o.flag.startswith("-?"):
            arg = args[o.index]
            return arg[:arg.index(o.flag[2]) + 1] + "..."
    return None


BOOLEAN_LONG = frozenset(("--detach", "--interactive", "--tty"))


def uses(args: list[str], flags: tuple[str, ...]) -> str | None:
    """The first argument that sets one of `flags` (`-u`, `--user`...) in any spelling, or None. `--tty=false` and its like do not set anything."""
    return next((args[o.index] for o in options(args) if o.flag in flags and not (o.flag in BOOLEAN_LONG and o.value == "false")), None)


# `docker run` must stay in the foreground, with the container's exit code and a /output that nobody still writes to (the suite snapshots it when the container is gone).
FOREGROUND_BREAKERS = ("-d", "--detach", "-a", "--attach", "--restart", "--cidfile")


def foreground_problem(args: list[str], shown=lambda arg: arg) -> str | None:
    """Why these arguments would make a conversion run in the background, conflict with `--rm`, fail on the second conversion, or not attach stdin; None if they are fine.
    Checked on a docker 2x: -d returns the container id at once; --restart is "conflicting options: cannot specify both --restart and --rm" (exit 125); the second run
    with the same --cidfile is "container ID file found" (exit 125); -i with -t is "cannot attach stdin to a TTY-enabled container" (exit 1). Either of -i and -t alone is fine.
    `shown` prepares an argument for the message (the caller masks environment values: an argument like `-dteTOKEN=x` must not be echoed)."""
    if bad := uses(args, FOREGROUND_BREAKERS):
        return (f"{shown(bad)}: the suite needs `docker run` in the foreground (it reads /output when the container exits, runs about thirty conversions and gives each "
                "its own --rm container); -d/--detach, -a/--attach, --restart and --cidfile are refused")
    interactive, tty = uses(args, ("-i", "--interactive")), uses(args, ("-t", "--tty"))
    if interactive and tty:
        both = "-i and -t together" if interactive == tty else f"{shown(interactive)} together with {shown(tty)}"
        return f"{both}: docker cannot attach stdin to a terminal when the suite's stdin is /dev/null; use either one, not both"
    return None
