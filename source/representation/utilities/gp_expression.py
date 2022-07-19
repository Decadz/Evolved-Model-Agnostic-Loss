from deap import creator
from deap import base
from deap import gp

import torch


# ==================================================
# Arithmetic Operators:
# ==================================================

def add(arg1, arg2):
    return torch.add(arg1, arg2)


def sub(arg1, arg2):
    return torch.sub(arg1, arg2)


def mul(arg1, arg2):
    return torch.mul(arg1, arg2)


def div(arg1, arg2):
    # Analytic quotient operator which is a smooth deterministic approximation of division.
    return torch.div(arg1, torch.sqrt(torch.add(1, torch.square(arg2))))


# ==================================================
# Unary Operators:
# ==================================================

def sign(arg):
    return torch.sign(arg)


def square(arg):
    return torch.pow(arg, 2)


def abs(arg):
    return torch.abs(arg)


def log(arg):
    return torch.log(torch.clamp(torch.abs(arg), min=1e-7))


def sqrt(arg):
    return torch.sqrt(torch.clamp(torch.abs(arg), min=1e-7))


# ==================================================
# Hyperbolic Functions:
# ==================================================


def tanh(arg):
    return torch.tanh(arg)


# ==================================================
# Set Operators:
# ==================================================

def min(arg1, arg2):
    return torch.min(arg1, arg2)


def max(arg1, arg2):
    return torch.max(arg1, arg2)


# ==================================================
# Defining DEAP function set:
# ==================================================

# NOTE: Ensure arithmetic operators are defined first in
#       the function set at all times to ensure correction
#       procedure works as intended.

# Creating a new primitive node set with two input terminals y and f(x).
pset = gp.PrimitiveSet("main", 2)

# Defining the arithmetic operators.
pset.addPrimitive(add, 2, name="add")
pset.addPrimitive(sub, 2, name="sub")
pset.addPrimitive(mul, 2, name="mul")
pset.addPrimitive(div, 2, name="div")

# Defining the unary operators.
pset.addPrimitive(sign, 1, name="sign")
pset.addPrimitive(square, 1, name="square")
pset.addPrimitive(abs, 1, name="abs")
pset.addPrimitive(log, 1, name="log")
pset.addPrimitive(sqrt, 1, name="sqrt")

# Defining the hyperbolic functions.
pset.addPrimitive(tanh, 1, name="tanh")

# Defining the set operators.
pset.addPrimitive(min, 2, name="min")
pset.addPrimitive(max, 2, name="max")

# Adding constants to the terminal set.
pset.addEphemeralConstant("+1", lambda: 1)
pset.addEphemeralConstant("-1", lambda: -1)

# Defining the DEAP GP expression tree object.
creator.create("Minimize", base.Fitness, weights=(-1.0,))
creator.create("Expression", gp.PrimitiveTree, fitness=creator.Minimize, pset=pset)

# Creating a dictionary for convenient operation lookup.
operations = {

    # PyTorch arithmetic function nodes.
    "add": add, "sub": sub, "mul": mul, "div": div,

    # PyTorch unary function nodes.
    "sign": sign, "square": square, "abs": abs,
    "sqrt": sqrt, "log": log,

    # PyTorch hyperbolic function nodes.
    "tanh": tanh,

    # PyTorch set operators nodes.
    "min": min, "max": max,

}
