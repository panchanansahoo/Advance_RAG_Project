"""
Pandas Agent for executing LLM-generated code against CSV/Excel files.

Phase 4 component.
"""

from __future__ import annotations

import logging
import ast
import traceback
import asyncio
from typing import Dict, Any, List

import pandas as pd
from backend.generation.llm import get_llm
from backend.config import get_settings

logger = logging.getLogger(__name__)


class PandasAgent:
    """
    Executes analytical queries on tabular data files (CSV, Excel) by 
    having an LLM generate Python Pandas code, running the code safely, 
    and then summarizing the result.
    """

    def __init__(self):
        self.settings = get_settings()

    async def run(self, query: str, file_paths: List[str]) -> str:
        """
        Execute a user query against a list of file paths (CSV/XLSX).
        """
        if not file_paths:
            return "No tabular files provided to analyze."

        # Read samples to provide schema context to the LLM
        schemas = {}
        for idx, path in enumerate(file_paths):
            try:
                if path.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(path, nrows=5)
                    df_full = pd.read_excel(path)
                else:
                    df = pd.read_csv(path, nrows=5)
                    df_full = pd.read_csv(path)
                schemas[f"df_{idx}"] = {
                    "path": path,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
                    "sample": df.head(2).to_dict(orient="records"),
                    "total_rows": len(df_full),
                }
            except Exception as e:
                logger.warning("Failed to load schema for %s: %s", path, e)

        if not schemas:
            return "I couldn't read the data tables in the provided spreadsheet files. Please verify the file format and try again."

        # Step 1: Generate Code
        code = await self._generate_code(query, schemas)
        if not code:
            return "I couldn't determine how to calculate the answer from this spreadsheet. Please try rephrasing your question."

        # Step 2: Execute Code
        result = await self._execute_code(code, schemas)

        # Step 2.5: Retry with error feedback if first execution failed
        if isinstance(result, str) and result.startswith(("Execution Error:", "Syntax Error")):
            logger.info("First code execution failed, retrying with error feedback")
            code_retry = await self._generate_code(
                query, schemas,
                error_feedback=f"The previous code failed with: {result}. Please fix and try again.",
            )
            if code_retry:
                result = await self._execute_code(code_retry, schemas)

        # Step 3: Summarize Result
        final_answer = await self._summarize_result(query, result)
        return final_answer

    async def _generate_code(self, query: str, schemas: Dict[str, Any], error_feedback: str = "") -> str:
        """Ask the LLM to generate Pandas code."""
        llm = get_llm()
        
        schema_desc = ""
        for df_name, info in schemas.items():
            schema_desc += f"\nDataframe Name: `{df_name}` (loaded from {info['path']})\n"
            schema_desc += f"Columns: {info['columns']}\n"
            schema_desc += f"Types: {info['dtypes']}\n"
            schema_desc += f"Total Rows: {info.get('total_rows', 'unknown')}\n"
            schema_desc += f"Sample Row: {info['sample']}\n"

        system_prompt = (
            "You are an expert Python data analyst. The user has asked a question about some tabular data files. "
            "I have already loaded these files into memory as Pandas DataFrames. "
            f"Here are the schemas for the available DataFrames:\n{schema_desc}\n\n"
            "Your task is to write a Python script using pandas to answer the user's question. "
            "Rules:\n"
            "1. The dataframes are already loaded and named df_0, df_1, etc. Do NOT write `pd.read_csv` or `pd.read_excel`.\n"
            "2. Compute the answer and assign the final result (a number, string, or small summary dataframe) "
            "to a variable named `result`.\n"
            "3. Do NOT print the result.\n"
            "4. Return ONLY valid Python code inside a ```python block. No explanations.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]

        if error_feedback:
            messages.append({"role": "user", "content": error_feedback})

        try:
            response = await llm.generate(messages, temperature=0.0, max_tokens=300)
            
            # Extract code block
            if "```python" in response:
                code = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                code = response.split("```")[1].strip()
            else:
                code = response.strip()
                
            logger.info("Generated Pandas code:\n%s", code)
            return code
        except Exception as e:
            logger.error("Failed to generate pandas code: %s", e)
            return ""

    async def _execute_code(self, code: str, schemas: Dict[str, Any]) -> str:
        """Safely execute the generated Pandas code with hardened security checks."""

        # ── Security Check: comprehensive AST analysis ──────
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Block unauthorized imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    allowed_modules = {"pandas", "pd", "numpy", "np", "math", "datetime"}
                    for alias in node.names:
                        module_name = alias.name.split(".")[0]
                        if module_name not in allowed_modules:
                            return (
                                f"Execution Error: Import of '{alias.name}' is not allowed "
                                "for security reasons."
                            )
                    if isinstance(node, ast.ImportFrom) and node.module:
                        root_module = node.module.split(".")[0]
                        if root_module not in allowed_modules:
                            return (
                                f"Execution Error: Import from '{node.module}' is not allowed "
                                "for security reasons."
                            )

                # Block dangerous function calls
                if isinstance(node, ast.Call):
                    func = node.func
                    dangerous_funcs = {
                        "eval", "exec", "compile", "__import__",
                        "open", "getattr", "setattr", "delattr",
                        "globals", "locals", "vars", "dir",
                        "breakpoint", "exit", "quit",
                    }
                    if isinstance(func, ast.Name) and func.id in dangerous_funcs:
                        return (
                            f"Execution Error: Call to '{func.id}()' is not allowed "
                            "for security reasons."
                        )
                    # Block method calls like os.system(), subprocess.run(), file-writing
                    if isinstance(func, ast.Attribute):
                        dangerous_methods = {
                            "system", "popen", "run", "call", "check_output",
                            # File-writing methods (data exfiltration prevention)
                            "to_csv", "to_excel", "to_pickle", "to_parquet",
                            "to_hdf", "to_feather", "to_json",
                        }
                        if func.attr in dangerous_methods:
                            return (
                                f"Execution Error: Call to '.{func.attr}()' is not allowed "
                                "for security reasons."
                            )

                # Block dangerous attribute access
                if isinstance(node, ast.Attribute):
                    dangerous_attrs = {
                        "__subclasses__", "__bases__", "__globals__",
                        "__code__", "__builtins__", "__class__",
                        "__mro__", "__dict__",
                    }
                    if node.attr in dangerous_attrs:
                        return (
                            f"Execution Error: Access to '{node.attr}' is not allowed "
                            "for security reasons."
                        )

        except SyntaxError as e:
            return f"Syntax Error in generated code: {e}"

        # Load the full dataframes into the local execution context
        exec_locals = {}
        for df_name, info in schemas.items():
            if info["path"].lower().endswith((".xlsx", ".xls")):
                exec_locals[df_name] = pd.read_excel(info["path"])
            else:
                exec_locals[df_name] = pd.read_csv(info["path"])

        import numpy as np

        exec_globals = {
            "pd": pd,
            "np": np,
            "__builtins__": {
                "range": range, "len": len, "sum": sum, "min": min, "max": max,
                "abs": abs, "round": round, "int": int, "float": float, "str": str,
                "list": list, "dict": dict, "set": set, "bool": bool,
                "tuple": tuple, "sorted": sorted, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter, "reversed": reversed,
                "True": True, "False": False, "None": None,
                "isinstance": isinstance, "type": type, "print": lambda *a, **k: None,
                "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
                "KeyError": KeyError, "IndexError": IndexError,
            }
        }

        def run_exec():
            try:
                exec(code, exec_globals, exec_locals)
                return exec_locals.get("result", "Execution successful but `result` variable was not set.")
            except Exception as e:
                logger.error("Pandas code execution failed: %s\n%s", e, traceback.format_exc())
                return f"Execution Error: {e}"

        try:
            # Run in a separate thread to avoid blocking the async event loop
            loop = asyncio.get_running_loop()
            result_obj = await asyncio.wait_for(
                loop.run_in_executor(None, run_exec),
                timeout=self.settings.pandas_execution_timeout
            )
            
            # Format output
            if isinstance(result_obj, pd.DataFrame):
                return result_obj.to_markdown()
            elif isinstance(result_obj, pd.Series):
                return result_obj.to_frame().to_markdown()
            else:
                return str(result_obj)
                
        except asyncio.TimeoutError:
            return f"Execution Timeout: Code took longer than {self.settings.pandas_execution_timeout} seconds."
        except Exception as e:
            logger.error("Unexpected error during pandas execution: %s", e)
            return f"Unexpected Error: {e}"

    async def _summarize_result(self, query: str, execution_result: str) -> str:
        """Formulate a natural language answer based on the execution result."""
        llm = get_llm()
        
        system_prompt = (
            "You are a helpful data analyst. The user asked a question about spreadsheet data. "
            "Based on the execution result provided, formulate a clear, concise, and natural language answer to the user's question.\n\n"
            "RULES:\n"
            "- Never mention programming terms, Python, code snippets, syntax, libraries, or stack traces.\n"
            "- If an error or limitation occurred, explain politely in simple non-technical words and advise how to adjust the question.\n"
            "- If the result contains calculated numbers or tables, present them clearly."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Query: {query}\n\nExecution Result:\n{execution_result}"}
        ]

        try:
            return await llm.generate(messages, temperature=0.3, max_tokens=300)
        except Exception as e:
            logger.error("Failed to summarize result: %s", e)
            from backend.utils.error_sanitizer import sanitize_error_message
            return sanitize_error_message(
                execution_result,
                default_fallback="I was unable to complete the calculation on this spreadsheet data. Please check your question or column names and try again.",
            )
