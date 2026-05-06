import concurrent.futures
import io
import resource
import signal
import sys
import threading
import traceback

from .model import RunPythonCodeResponse
from .utils import load_config


# Load sandbox configuration
cfg = load_config()
sandbox_cfg = cfg['sandbox']
TIMEOUT_SECONDS = sandbox_cfg.get('timeout_seconds', 30)
MEMORY_LIMIT_MB = sandbox_cfg.get('memory_limit_mb', 512)
CPU_TIME_LIMIT_SECONDS = sandbox_cfg.get('cpu_time_limit_seconds', 15)
BLOCKED_MODULES = set(sandbox_cfg.get('blocked_modules', []))


TOOL_API_PREAMBLE = '''
import requests
api_url = "<API-ENDPOINT>"

def list_directory(id: int, prefix: str = '/*', depth: int = 1) -> dict:
    url = f'{api_url}/api/directory/{id}'
    params = {'prefix': prefix, 'depth': depth}
    response = requests.get(url=url, params=params)
    return response.json()

def read_text_file(id: int, path: str, head: int | None = None, tail: int | None = None) -> dict:
    url = f'{api_url}/api/text_file/{id}'
    params = {'path': path, 'head': head, 'tail': tail}
    return requests.get(url=url, params=params).json()

def read_binary_file(id: int, path: str) -> dict:
    url = f'{api_url}/api/binary_file/{id}'
    params = {'path': path}
    return requests.get(url=url, params=params).json()

'''


def _timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError(f"Code execution exceeded {TIMEOUT_SECONDS} second time limit")


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Custom import function that blocks dangerous modules.
    Allows safe modules like requests, json, math, etc.
    """
    # Check if module is in blocked list
    if name in BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    
    # Check if trying to access dangerous submodules
    top_level = name.split('.')[0]
    if top_level in BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    
    # Use the builtin __import__
    return __import__(name, globals, locals, fromlist, level)


def _create_restricted_builtins():
    """
    Create a restricted set of builtins that disallow dangerous operations.
    """
    # Start with a minimal safe set
    safe_builtins = {
        # Safe built-in functions
        'abs': abs,
        'all': all,
        'any': any,
        'ascii': ascii,
        'bin': bin,
        'bool': bool,
        'bytearray': bytearray,
        'bytes': bytes,
        'chr': chr,
        'complex': complex,
        'dict': dict,
        'dir': dir,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'format': format,
        'frozenset': frozenset,
        'hex': hex,
        'int': int,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'iter': iter,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'print': print,
        'range': range,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'type': type,
        'zip': zip,
        # Safe constants
        'True': True,
        'False': False,
        'None': None,
        'NotImplemented': NotImplemented,
        'Ellipsis': Ellipsis,
        # Safe exceptions
        'BaseException': BaseException,
        'Exception': Exception,
        'ArithmeticError': ArithmeticError,
        'AssertionError': AssertionError,
        'AttributeError': AttributeError,
        'BlockingIOError': BlockingIOError,
        'BrokenPipeError': BrokenPipeError,
        'BufferError': BufferError,
        'BytesWarning': BytesWarning,
        'ChildProcessError': ChildProcessError,
        'ConnectionError': ConnectionError,
        'ConnectionAbortedError': ConnectionAbortedError,
        'ConnectionRefusedError': ConnectionRefusedError,
        'ConnectionResetError': ConnectionResetError,
        'DeprecationWarning': DeprecationWarning,
        'EOFError': EOFError,
        'EnvironmentError': EnvironmentError,
        'FileExistsError': FileExistsError,
        'FileNotFoundError': FileNotFoundError,
        'FloatingPointError': FloatingPointError,
        'FutureWarning': FutureWarning,
        'GeneratorExit': GeneratorExit,
        'IndentationError': IndentationError,
        'IndexError': IndexError,
        'InterruptedError': InterruptedError,
        'IsADirectoryError': IsADirectoryError,
        'KeyError': KeyError,
        'KeyboardInterrupt': KeyboardInterrupt,
        'LookupError': LookupError,
        'MemoryError': MemoryError,
        'ModuleNotFoundError': ModuleNotFoundError,
        'NameError': NameError,
        'NotADirectoryError': NotADirectoryError,
        'NotImplementedError': NotImplementedError,
        'OSError': OSError,
        'OverflowError': OverflowError,
        'PendingDeprecationWarning': PendingDeprecationWarning,
        'PermissionError': PermissionError,
        'RecursionError': RecursionError,
        'ReferenceError': ReferenceError,
        'ResourceWarning': ResourceWarning,
        'RuntimeError': RuntimeError,
        'RuntimeWarning': RuntimeWarning,
        'StopAsyncIteration': StopAsyncIteration,
        'StopIteration': StopIteration,
        'SyntaxError': SyntaxError,
        'SyntaxWarning': SyntaxWarning,
        'SystemError': SystemError,
        'SystemExit': SystemExit,
        'TabError': TabError,
        'TimeoutError': TimeoutError,
        'TypeError': TypeError,
        'UnboundLocalError': UnboundLocalError,
        'UnicodeError': UnicodeError,
        'UnicodeDecodeError': UnicodeDecodeError,
        'UnicodeEncodeError': UnicodeEncodeError,
        'UnicodeTranslateError': UnicodeTranslateError,
        'UnicodeWarning': UnicodeWarning,
        'UserWarning': UserWarning,
        'ValueError': ValueError,
        'Warning': Warning,
        'ZeroDivisionError': ZeroDivisionError,
        # Safe utility
        'compile': compile,
        'eval': eval,
        'hash': hash,
        'id': id,
        'len': len,
        'vars': vars,
        '__import__': _restricted_import,
    }
    return safe_builtins


def _execute_code_in_namespace(code: str, namespace: dict, output_buffer: io.StringIO) -> str:
    """Execute code and return captured output."""
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    try:
        exec(code, namespace)
        return output_buffer.getvalue()
    finally:
        sys.stdout = original_stdout


import ast
from typing import Optional

def _has_explicit_print(tree: ast.AST) -> bool:
    """Return True if there is any call to a name 'print' in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                return True
    return False

def _collect_name_nodes(target: ast.AST, out: list):
    """Collect ast.Name nodes from a target (recursively for tuples)."""
    if isinstance(target, ast.Name):
        out.append(ast.Name(id=target.id, ctx=ast.Load()))
    elif isinstance(target, ast.Tuple):
        for elt in target.elts:
            _collect_name_nodes(elt, out)
    # ignore other complex targets (attributes, subscripts); they could be added if desired


def _add_implicit_print(code: str, require_no_print: bool = True) -> str:
    """
    Return source with an implicit print appended if appropriate.

    Parameters:
      - code: python source (a block) as string
      - require_no_print: if True (default) and any explicit print(...) exists anywhere,
                          the function returns the original code unchanged.

    Behavior decisions:
      - For Assign with multiple targets (e.g. `a = b = 1`) we print the first simple Name target found.
      - For complex targets (attributes, subscripts) we currently don't auto-print them.
    """
    tree = ast.parse(code)

    # Optionally bail out if there is already an explicit print call somewhere
    if require_no_print and _has_explicit_print(tree):
        return code

    if not tree.body:
        return code

    last = tree.body[-1]

    # If last is an expression (e.g. `2 + 3`, `y`, a function call other than print),
    # wrap that expression in print(...)
    if isinstance(last, ast.Expr):
        # if it's already a print(...) call, do nothing
        if isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name) and last.value.func.id == "print":
            return code

        # create: print(<last.value>)
        print_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[last.value],
                keywords=[]
            )
        )
        tree.body.append(print_call)

    # For all other kinds of final statements (def, class, import, control flow, etc.), do nothing
    else:
        return code

    # Convert AST back to source. Prefer ast.unparse (py3.9+); fall back to astor if needed.
    try:
        new_source = ast.unparse(tree)
    except AttributeError:
        # ast.unparse not available -> try astor
        try:
            import astor
        except ImportError:
            raise RuntimeError("ast.unparse not available and astor not installed; cannot unparse AST")
        new_source = astor.to_source(tree)

    return new_source


def run_code(code: str) -> RunPythonCodeResponse:
    """
    Execute Python code in an isolated namespace with security safeguards:
    - Timeout protection (works in both main and worker threads)
    - Memory limit
    - CPU time limit
    - Restricted module imports
    - Restricted builtins
    
    Each execution has a clean slate - no variables persist between calls.
    """
    # Create isolated namespace for this execution
    namespace = {
        '__builtins__': _create_restricted_builtins(),
        '__name__': '__sandbox__',
    }
    
    # Capture stdout
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    original_alarm = None
    use_signal_timeout = threading.current_thread() is threading.main_thread()

    # add preamble and add print for last line if applicable
    code = TOOL_API_PREAMBLE + _add_implicit_print(code)
    
    try:
        # Set resource limits (memory and CPU time)
        try:
            # Memory limit: convert MB to bytes
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_MB * 1024 * 1024, MEMORY_LIMIT_MB * 1024 * 1024))
            # CPU time limit (soft limit only, won't crash on exceed)
            resource.setrlimit(resource.RLIMIT_CPU, (CPU_TIME_LIMIT_SECONDS, CPU_TIME_LIMIT_SECONDS + 5))
        except Exception as e:
            # Resource limits may not be available on all systems
            pass
        
        # Execute code with timeout
        if use_signal_timeout:
            # Main thread: use signal-based timeout
            try:
                original_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(TIMEOUT_SECONDS)
            except ValueError:
                # signal.signal() failed, fall back to thread-based timeout
                use_signal_timeout = False
            
            if use_signal_timeout:
                # Redirect stdout to capture print statements
                sys.stdout = output_buffer
                
                # Execute code in isolated namespace
                exec(code, namespace)
                
                # Cancel the alarm
                signal.alarm(0)
                
                captured_output = output_buffer.getvalue()
                response = RunPythonCodeResponse(
                    status='success',
                    output=captured_output,
                    error=None
                )
                return response
        
        # Worker thread: use ThreadPoolExecutor with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _execute_code_in_namespace,
                code,
                namespace,
                output_buffer
            )
            try:
                captured_output = future.result(timeout=TIMEOUT_SECONDS)
                response = RunPythonCodeResponse(
                    status='success',
                    output=captured_output,
                    error=None
                )
            except concurrent.futures.TimeoutError:
                captured_output = output_buffer.getvalue()
                response = RunPythonCodeResponse(
                    status='error',
                    output=captured_output,
                    error=f"Code execution exceeded {TIMEOUT_SECONDS} second time limit"
                )
    
    except TimeoutError as e:
        captured_output = output_buffer.getvalue()
        response = RunPythonCodeResponse(
            status='error',
            output=captured_output,
            error=str(e)
        )
    
    except Exception as e:
        if use_signal_timeout:
            signal.alarm(0)  # Cancel alarm on any exception
        captured_output = output_buffer.getvalue()
        error_traceback = traceback.format_exc()
        response = RunPythonCodeResponse(
            status='error',
            output=captured_output,
            error=error_traceback
        )
    
    finally:
        # Always restore state
        if use_signal_timeout:
            signal.alarm(0)  # Cancel any pending alarm
        sys.stdout = original_stdout
        if use_signal_timeout and original_alarm is not None:
            signal.signal(signal.SIGALRM, original_alarm)
        output_buffer.close()

    return response
