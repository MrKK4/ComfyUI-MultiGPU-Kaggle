"""Static guard for the Triton launchers in turing_attention.py.

Triton launchers are called as `_KERNEL[grid](arg, arg, ...)`; nothing checks the argument list
until the kernel actually launches on a GPU, so an arity slip (an extra positional, a stride
forgotten, a constexpr passed positionally) costs a round trip to Kaggle to find. This walks the
AST of the module, matches every launcher call against its kernel definition, and reports the
mismatch. It needs no torch, no GPU and no triton — run it after every edit to the kernels.

    python check_kernels.py [file.py]      # exits 1 on any mismatch
"""
import ast
import sys

SRC = "turing_attention.py"
# Triton consumes these itself; they are not parameters of the jitted function
LAUNCH_KWARGS = {"num_warps", "num_stages", "num_ctas", "maxnreg", "grid"}


def kernel_defs(tree):
    """name -> (param names, defaulted/constexpr names) for plain positional-arg functions."""
    defs = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.args.kwonlyargs:
            continue
        if node.args.vararg or node.args.kwarg:
            continue
        names = [a.arg for a in node.args.args]
        n_default = len(node.args.defaults)
        defs[node.name] = (names, set(names[len(names) - n_default:]) if n_default else set())
    return defs


def resolve_aliases(tree):
    """_FA_I8_K = _FA_I8  ->  {_FA_I8_K: _FA_I8}"""
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    alias[target.id] = node.value.id
    return alias


def launcher_calls(tree, names):
    """(lineno, called_name, n_positional, [keyword names]) for every `_KERNEL[grid](...)`."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Subscript) and isinstance(fn.value, ast.Name)):
            continue
        if fn.value.id not in names:
            continue
        calls.append((node.lineno, fn.value.id, len(node.args),
                      [k.arg for k in node.keywords if k.arg]))
    return calls


def check(path=SRC):
    tree = ast.parse(open(path).read())
    defs = kernel_defs(tree)
    alias = {k: v for k, v in resolve_aliases(tree).items() if k.endswith("_K")}
    problems, checked = [], 0
    for lineno, called, n_pos, kw_all in launcher_calls(tree, set(alias) | set(defs)):
        target = alias.get(called, called)
        if target not in defs:
            problems.append(f"line {lineno}: launcher {called}[...] has no kernel definition")
            continue
        params, defaulted = defs[target]
        kw_names = [k for k in kw_all if k not in LAUNCH_KWARGS]
        checked += 1
        if n_pos + len(kw_names) != len(params):
            problems.append(f"line {lineno}: {called} called with {n_pos} positional + "
                            f"{len(kw_names)} kernel keyword = {n_pos + len(kw_names)} args, "
                            f"kernel takes {len(params)}")
            continue
        clash = [params[i] for i in range(n_pos) if params[i] in defaulted]
        if clash:
            problems.append(f"line {lineno}: {called} passes {clash} positionally, but they are "
                            f"constexpr/defaulted — pass them by keyword")
        unknown = [k for k in kw_names if k not in params]
        if unknown:
            problems.append(f"line {lineno}: {called} passed unknown keyword(s) {unknown}")
    for p in problems:
        print("FAIL " + p)
    print(f"{checked} launcher call(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(check(*sys.argv[1:]))
