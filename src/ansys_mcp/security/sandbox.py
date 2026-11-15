"""
Security Governance & AST Sandbox Engine
----------------------------------------
Implements 3-Level execution security for AI-generated scripts.

Two layers protect the server:

* a static AST policy that rejects unsafe imports, system calls and every route
  to Python internals (dunder attributes, ``"__"`` strings, ``str.format``
  attribute lookups), and
* execution in a separate, isolated interpreter (``python -I``) with a time limit
  and capped output.  The script's ``print`` goes to a buffer, never to the
  server's stdout, which is the MCP transport; a runaway loop is killed instead
  of freezing the server.
"""

import ast
import json
import os
import subprocess
import sys
from typing import Dict, Any, List, Set, Optional
from pydantic import BaseModel, Field

#: Wall-clock limit for one script, in seconds.
EXECUTION_TIMEOUT_S = 10

#: What a script may import, by security level.
ALLOWED_MODULES = {
    2: {"math", "cmath", "statistics"},
    3: {"math", "cmath", "statistics", "numpy", "json", "re", "itertools", "functools"},
}

MAX_OUTPUT_ITEMS = 50
MAX_OUTPUT_CHARS = 2000
MAX_PRINT_CHARS = 10000


class SecurityCheckResult(BaseModel):
    is_safe: bool
    security_level: int
    forbidden_calls: List[str] = Field(default_factory=list)
    forbidden_imports: List[str] = Field(default_factory=list)
    forbidden_syntax: List[str] = Field(default_factory=list)
    risk_score: float = 0.0
    verdict: str


class AstSecurityValidator(ast.NodeVisitor):
    """Parses AST nodes to detect blacklisted imports, system calls and internals access."""

    FORBIDDEN_MODULES: Set[str] = {
        "os", "sys", "subprocess", "shutil", "socket", "http",
        "requests", "urllib", "importlib", "builtins", "ctypes",
        "pty", "commands", "posix", "nt", "winreg", "msvcrt",
        "pathlib", "io", "pickle", "marshal", "inspect", "gc", "threading",
        "multiprocessing", "asyncio", "signal", "tempfile", "glob",
    }

    FORBIDDEN_CALLS: Set[str] = {
        "eval", "exec", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "system", "popen", "spawn",
        "open", "input", "vars", "dir", "breakpoint", "help", "memoryview",
        "format_map", "format",
    }

    def __init__(self, allowed_modules: Optional[Set[str]] = None):
        self.allowed_modules = allowed_modules or set()
        self.forbidden_imports: List[str] = []
        self.forbidden_calls: List[str] = []
        self.forbidden_syntax: List[str] = []

    def _check_module(self, name: str) -> None:
        base = name.split(".")[0]
        if base in self.FORBIDDEN_MODULES or base not in self.allowed_modules:
            self.forbidden_imports.append(name)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._check_module(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level or not node.module:
            self.forbidden_imports.append("." * node.level + (node.module or ""))
        else:
            self._check_module(node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                self.forbidden_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_CALLS:
                self.forbidden_calls.append(node.func.attr)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Every escape from a restricted namespace goes through an underscore
        # attribute (__class__, __subclasses__, __globals__, _private).
        if node.attr.startswith("_"):
            self.forbidden_syntax.append("attribute '%s'" % node.attr)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id.startswith("__"):
            self.forbidden_syntax.append("name '%s'" % node.id)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        # "{0.__class__}".format(x) reads attributes without an Attribute node.
        if isinstance(node.value, str) and "__" in node.value:
            self.forbidden_syntax.append("string containing '__'")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        self.forbidden_syntax.append("global statement")

    def visit_Nonlocal(self, node: ast.Nonlocal):
        self.forbidden_syntax.append("nonlocal statement")


#: Runs inside the child interpreter.  Reads {"code","level","job_dir","modules"} from stdin,
#: executes with a minimal namespace and prints one JSON document on the real stdout.
_RUNNER = r'''
import builtins, io, json, math, sys

request = json.loads(sys.stdin.read())
real_stdout = sys.stdout
buffer = io.StringIO()
allowed = set(request["modules"])

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in allowed:
        raise ImportError("Modül izinli değil: " + name)
    return __import__(name, globals, locals, fromlist, level)

def safe_print(*args, **kwargs):
    kwargs.pop("file", None)
    print(*args, file=buffer, **kwargs)

names = ["range", "len", "abs", "min", "max", "round", "float", "int", "str", "list", "dict",
         "set", "sum", "enumerate", "zip", "sorted", "reversed", "map", "filter", "bool", "tuple",
         "any", "all", "pow", "divmod", "isinstance", "True", "False", "None", "ValueError",
         "ZeroDivisionError", "Exception", "TypeError", "KeyError", "IndexError"]
safe_builtins = {n: getattr(builtins, n) for n in names if hasattr(builtins, n)}
safe_builtins["print"] = safe_print
safe_builtins["__import__"] = safe_import
namespace = {"__builtins__": safe_builtins, "math": math, "JOB_DIR": request["job_dir"]}
local_names = {}
try:
    exec(compile(request["code"], "<sandbox>", "exec"), namespace, local_names)
    out = {"status": "success",
           "values": {k: str(v) for k, v in local_names.items() if not k.startswith("_")}}
except BaseException as exc:
    out = {"status": "runtime_error", "message": type(exc).__name__ + ": " + str(exc)}
out["printed"] = buffer.getvalue()
real_stdout.write("\n@@RESULT@@" + json.dumps(out))
'''


class SandboxEngine:
    """Enforces 3-tier security governance."""

    @classmethod
    def validate_script_security(
        cls,
        script_code: str,
        security_level: int = 2
    ) -> SecurityCheckResult:
        """
        Validates Python script against AST security policies.
        Level 1: Only predefined parametric actions allowed.
        Level 2: AST sandbox (math/statistics only, no internals access).
        Level 3: Expert mode with user confirmation (adds numpy, json, re, itertools).
        """
        if security_level == 1:
            return SecurityCheckResult(
                is_safe=False,
                security_level=1,
                verdict="Level 1: Keyfi kod çalıştırma tamamen devre dışıdır. Standart MCP Tool'larını kullanınız."
            )
        if security_level not in ALLOWED_MODULES:
            return SecurityCheckResult(
                is_safe=False,
                security_level=security_level,
                verdict="Geçersiz güvenlik seviyesi: %r (1, 2 veya 3 olmalı)." % (security_level,),
            )

        try:
            tree = ast.parse(script_code)
        except SyntaxError as se:
            return SecurityCheckResult(
                is_safe=False,
                security_level=security_level,
                verdict=f"Sözdizimi Hatası (SyntaxError): {str(se)}"
            )

        validator = AstSecurityValidator(ALLOWED_MODULES[security_level])
        validator.visit(tree)

        violations = (
            validator.forbidden_imports or validator.forbidden_calls or validator.forbidden_syntax
        )
        if violations:
            risk = 1.0
            is_safe = False
            verdict = (
                f"GÜVENLİK İHLALİ: Yasaklı kütüphane, sistem çağrısı veya iç yapı erişimi tespit edildi! "
                f"Yasaklı Importlar: {validator.forbidden_imports}, Çağrılar: {validator.forbidden_calls}, "
                f"Sözdizimi: {validator.forbidden_syntax}"
            )
        else:
            risk = 0.0
            is_safe = True
            verdict = "AST Güvenlik Denetimi Başarılı. Kod güvenli parametrik sandbox içerisinde çalıştırılabilir."

        return SecurityCheckResult(
            is_safe=is_safe,
            security_level=security_level,
            forbidden_imports=validator.forbidden_imports,
            forbidden_calls=validator.forbidden_calls,
            forbidden_syntax=validator.forbidden_syntax,
            risk_score=risk,
            verdict=verdict
        )

    @classmethod
    def execute_sandboxed_code(
        cls,
        script_code: str,
        job_dir: str,
        security_level: int = 2,
        confirmation_token: Optional[str] = None,
        timeout_s: float = EXECUTION_TIMEOUT_S,
    ) -> Dict[str, Any]:
        """
        Runs the script in a separate isolated interpreter with a time limit.
        """
        if security_level == 3:
            if confirmation_token != "CONFIRM_EXPERT_EXECUTION":
                return {
                    "status": "error",
                    "message": "Level 3 çalıştırma için 'CONFIRM_EXPERT_EXECUTION' onay belirteci zorunludur."
                }

        check = cls.validate_script_security(script_code, security_level)
        if not check.is_safe:
            return {
                "status": "blocked",
                "check_result": check.model_dump(),
                "message": check.verdict
            }

        request = json.dumps({
            "code": script_code,
            "level": security_level,
            "job_dir": job_dir,
            "modules": sorted(ALLOWED_MODULES[security_level]),
        })
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                input=request, capture_output=True, text=True, timeout=timeout_s,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "message": "Sandbox zaman aşımı: script %.0f saniyede bitmedi ve sonlandırıldı." % timeout_s,
            }

        marker = "@@RESULT@@"
        if marker not in proc.stdout:
            return {
                "status": "runtime_error",
                "message": "Sandbox sonuç üretmeden sonlandı: %s" % (proc.stderr.strip()[-300:] or "bilinmeyen hata"),
            }
        result = json.loads(proc.stdout.rsplit(marker, 1)[1])
        printed = (result.get("printed") or "")[:MAX_PRINT_CHARS]
        if result["status"] != "success":
            return {
                "status": "runtime_error",
                "message": f"Sandbox çalışma hatası: {result['message']}",
                "printed": printed,
            }
        values = result["values"]
        output = {k: v[:MAX_OUTPUT_CHARS] for k, v in list(values.items())[:MAX_OUTPUT_ITEMS]}
        return {
            "status": "success",
            "output": output,
            "printed": printed,
            "truncated": len(values) > MAX_OUTPUT_ITEMS,
            "security_verdict": check.verdict,
        }
