from fastmcp import FastMCP
import argparse
import math
import numpy as np
from scipy import stats
from sympy import symbols, solve, sympify, diff, integrate, oo, Sum
from typing import List, Tuple
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import integrate as sympy_integrate

# Create MCP Server
mcp = FastMCP("A server for complex mathematical calculations","1.0.0")



ALLOW_FUNCTION = {
    "math": math,
    "np": np,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "cot": lambda x: 1 / math.tan(x),
    "csc": lambda x: 1 / math.sin(x),
    "sec": lambda x: 1 / math.cos(x),
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "factorial": math.factorial,
    "gamma": math.gamma,
    "erf": math.erf,
    "erfc": math.erfc,
    "lgamma": math.lgamma,
    "degrees": math.degrees,
    "radians": math.radians,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "isqrt": math.isqrt,
    "prod": np.prod,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "clip": np.clip,
    "unique": np.unique,
    "sort": np.sort,
    "argsort": np.argsort,
    "argmax": np.argmax,
}


@mcp.tool
def calculate(expression: str) -> dict:
    """
    Evaluates a mathematical expression and returns the result.

    Supports basic operators (+, -, *, /, **, %), mathematical functions
    (sin, cos, tan, exp, log, log10, sqrt), and constants (pi, e).
    Uses a restricted evaluation context for safe execution.

    Args:
        expression: The mathematical expression to evaluate as a string.
                   Examples: "2 + 2", "sin(pi/4)", "sqrt(16) * 2", "log(100, 10)"

    Returns:
        On success: {"result": <calculated value>}
        On error: {"error": <error message>}

    Examples:
        >>> calculate("2 * 3 + 4")
        {'result': 10}
        >>> calculate("sin(pi/2)")
        {'result': 1.0}
        >>> calculate("sqrt(16)")
        {'result': 4.0}
        >>> calculate("invalid * expression")
        {'error': "name 'invalid' is not defined"}

    """
    try:
        # Safe evaluation of the expression
        result = eval(
            expression,
            {"__builtins__": {}},
            ALLOW_FUNCTION,
        )
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def solve_equation(equation: str) -> dict:
    """
    Solves an algebraic equation for x and returns all solutions.

    The equation must contain exactly one equality sign (=) and use a
    variable x. Can solve polynomial, trigonometric, and other equations
    supported by SymPy.

    Args:
        equation: The equation to solve as a string.
                 Format: '<left side> = <right side>'
                 Examples: "x**2 - 5*x + 6 = 0", "sin(x) = 0.5", "2*x + 3 = 7"

    Returns:
        On success: {"solutions": <list of solutions as string>}
        On error: {"error": <error message>}

    Examples:
        >>> solve_equation("x**2 - 5*x + 6 = 0")
        {'solutions': '[2, 3]'}
        >>> solve_equation("2*x + 3 = 7")
        {'solutions': '[2]'}
        >>> solve_equation("x = 0")
        {'solutions': '[0]'}
    """
    try:
        x = symbols("x")
        # Split the equation into left and right sides
        parts = equation.split("=")
        if len(parts) != 2:
            return {"error": "Equation must contain an '=' sign"}

        left = sympify(parts[0].strip())
        right = sympify(parts[1].strip())

        # Solve the equation
        solutions = solve(left - right, x)
        return {"solutions": str(solutions)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def differentiate(expression: str, variable: str = "x") -> dict:
    """
    Computes the derivative of a mathematical expression with respect to a variable.

    Supports polynomials, trigonometric functions, exponential functions,
    logarithms, and other functions supported by SymPy.

    Args:
        expression: The mathematical expression to differentiate as a string.
                   Examples: "x**2", "sin(x)", "exp(x)", "log(x)"
        variable: The variable with respect to which to differentiate. Default is "x".
                 Optionally, other variables can be specified.

    Returns:
        On success: {"result": <derivative as string>}
        On error: {"error": <error message>}
    """
    try:
        var = symbols(variable)
        expr = sympify(expression)
        result = diff(expr, var)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def integrate(expression: str, variable: str = "x") -> dict:
    """
    Computes the indefinite integral of a mathematical expression with respect to a variable.

    Supports polynomials, trigonometric functions, exponential functions,
    logarithms, and other functions supported by SymPy.

    Args:
        expression: The mathematical expression to integrate as a string.
                   Examples: "x**2", "sin(x)", "exp(x)", "1/x"
        variable: The variable with respect to which to integrate. Default is "x".
                 Optionally, other variables can be specified.

    Returns:
        On success: {"result": <integral as string>}
        On error: {"error": <error message>}


    """
    try:
        var = symbols(variable)
        expr = sympify(expression)
        result = sympy_integrate(expr, var)  # Use sympy_integrate instead of integrate
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def mean(data: List[float]) -> dict:
    """
    Computes the mean of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <mean value>}
        On error: {"error": <error message>}

    Examples:
        >>> mean([1, 2, 3, 4])
        {'result': 2.5}
        >>> mean([10, 20, 30])
        {'result': 20.0}
    """
    try:
        result = float(np.mean(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def variance(data: List[float]) -> dict:
    """
    Computes the variance of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <variance value>}
        On error: {"error": <error message>}
    """
    try:
        result = float(np.var(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def standard_deviation(data: List[float]) -> dict:
    """
    Computes the standard deviation of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <standard deviation value>}
        On error: {"error": <error message>}

    """
    try:
        result = float(np.std(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def median(data: List[float]) -> dict:
    """
    Computes the median of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <median value>}
        On error: {"error": <error message>}
    """
    try:
        result = float(np.median(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def mode(data: List[float]) -> dict:
    """
    Computes the mode of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <mode value>}
        On error: {"error": <error message>}

    """
    try:
        if not data:
            return {"error": "Cannot compute mode of empty array"}
        # Adjusted for newer SciPy versions
        mode_result = stats.mode(data, keepdims=False)
        return {"result": float(mode_result.mode)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def correlation_coefficient(data_x: List[float], data_y: List[float]) -> dict:
    """
    Computes the Pearson correlation coefficient between two lists of numbers.

    Args:
        data_x: The first list of numerical values.
        data_y: The second list of numerical values.

    Returns:
        On success: {"result": <correlation coefficient>}
        On error: {"error": <error message>}
    """
    try:
        result = np.corrcoef(data_x, data_y)[0, 1]
        return {"result": float(result)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def linear_regression(data: List[Tuple[float, float]]) -> dict:
    """
    Performs linear regression on a set of points and returns the slope and intercept.

    Args:
        data: A list of tuples, where each tuple contains (x, y) coordinates.

    Returns:
        On success: {"slope": <slope value>, "intercept": <intercept value>}
        On error: {"error": <error message>}

    """
    try:
        x = np.array([point[0] for point in data])
        y = np.array([point[1] for point in data])
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return {"slope": float(slope), "intercept": float(intercept)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def confidence_interval(data: List[float], confidence: float = 0.95) -> dict:
    """
    Computes the confidence interval for the mean of a dataset.

    Args:
        data: A list of numerical values.
        confidence: The confidence level (default is 0.95).

    Returns:
        On success: {"confidence_interval": <(lower_bound, upper_bound)>}
        On error: {"error": <error message>}

    """
    try:
        mean_value = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return {
            "confidence_interval": (
                float(mean_value - margin_of_error),
                float(mean_value + margin_of_error),
            )
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def matrix_addition(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> dict:
    """
    Adds two matrices.

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        On success: {"result": <resulting matrix>}
        On error: {"error": <error message>}

    """
    try:
        result = np.add(matrix_a, matrix_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def matrix_multiplication(
    matrix_a: List[List[float]], matrix_b: List[List[float]]
) -> dict:
    """
    Multiplies two matrices.

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        On success: {"result": <resulting matrix>}
        On error: {"error": <error message>}

    """
    try:
        result = np.dot(matrix_a, matrix_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def matrix_transpose(matrix: List[List[float]]) -> dict:
    """
    Transposes a matrix.

    Args:
        matrix: The matrix to transpose as a list of lists.

    Returns:
        On success: {"result": <transposed matrix>}
        On error: {"error": <error message>}

    """
    try:
        result = np.transpose(matrix).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def matrix_determinant(matrix: List[List[float]]) -> dict:
    """
    Multiplies two matrices.

    Args:
        matrix: The first vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}

    """
    try:
        result = np.linalg.det(matrix)
        return {"result": round(float(result), 10)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def vector_dot_product(vector_a: tuple[float], vector_b: tuple[float]) -> dict:
    """
    Multiplies two matrices.

    Args:
        vector_a: The first vector as a list of lists.
        vector_b: The second vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}
    """
    try:
        result = np.dot(vector_a, vector_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def vector_cross_product(vector_a: tuple[float], vector_b: tuple[float]) -> dict:
    """
    Multiplies two matrices.

    Args:
        vector_a: The first vector as a list of lists.
        vector_b: The second vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}
    """
    try:
        result = np.cross(vector_a, vector_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def vector_magnitude(vector: tuple[float]) -> dict:
    """
    Multiplies two matrices.

    Args:
        vector: The first vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}
    """
    try:
        result = np.linalg.norm(vector).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def plot_function(
    expression: str, start: int = -10, end: int = 10, step: int = 100
) -> dict:
    """
    Plots a graph of y = f(x).

    Args:
        x: The expression of function x as a string.

    Returns:
        On success: {"result": "Plot generated successfully."}
        On error: {"error": <error message>}

    """
    x = sp.Symbol("x")
    try:
        expression = sp.sympify(expression)
        f = sp.lambdify(x, expression, "numpy")
        x_values = np.linspace(start, end, step)
        y_values = f(x_values)
        fig, ax = plt.subplots()
        # Create quadrant graph
        ax.spines["left"].set_position("center")
        ax.spines["bottom"].set_position("center")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.plot(x_values, y_values)
        ax.set_xlabel("x", loc="right")
        ax.set_ylabel("f(x)", loc="top")
        ax.set_title(f"Graph of ${sp.latex(expression)}$")
        ax.grid(True)
        plt.show()
        return {"result": "Plot generated successfully."}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def summation(expression: str, start: int = 0, end: int = 10) -> dict:
    """
    Calculates the summation of a function from start to end.

    Args:
        expression: The expression of function x as a string.
        start: The starting value of the summation.
        end: The ending value of the summation.

    Returns:
        On success: {"result": <resulting summation>}
        On error: {"error": <error message>}

   
    """
    try:
        x = sp.Symbol("x")
        expr = sp.sympify(expression)
        summation = sp.Sum(expr, (x, start, end))
        result = summation.doit()
        return {"result": int(result) if result.is_integer else float(result)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def expand(expression: str) -> dict:
    """
    Expands an expression.

    Args:
        expression: The expression to expand as a string.

    Returns:
        On success: {"result": <expanded expression>}
        On error: {"error": <error message>}

   
    """
    try:
        x = sp.Symbol("x")
        expanded_expression = sp.expand(expression)
        return {"result": str(expanded_expression)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def factorize(expression: str) -> dict:
    """
    Factorizes an expression.

    Args:
        expression: The expression to factorize as a string.

    Returns:
        On success: {"result": <factored expression>}
        On error: {"error": <error message>}

    
    """
    try:
        x = sp.Symbol("x")
        factored_expression = sp.factor(expression)
        return {"result": str(factored_expression)}
    except Exception as e:
        return {"error": str(e)}


    
app = mcp.http_app()
