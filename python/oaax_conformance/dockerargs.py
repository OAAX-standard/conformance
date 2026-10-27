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
    """The first argument that is a cluster of short flags this reader cannot be sure about: it could hide an environment value or a flag we refuse."""
    return next((args[o.index] for o in options(args) if o.flag.startswith("-?")), None)


def uses(args: list[str], flags: tuple[str, ...]) -> str | None:
    """The first argument that sets one of `flags` (`-u`, `--user`...) in any spelling, or None."""
    return next((args[o.index] for o in options(args) if o.flag in flags), None)
