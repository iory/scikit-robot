def dummy_func(*args, **kwargs):
    message = "cvxopt is not installed. "\
              "Please install cvxopt by 'pip install cvxopt' "\
              "if wheel is released for your platform."
    raise ModuleNotFoundError(message)


try:
    from skrobot.optimizers import cvxopt_solver  # NOQA
    from skrobot.optimizers import quadprog_solver  # NOQA
except (ImportError, ModuleNotFoundError):
    cvxopt_solver = dummy_func
    quadprog_solver = dummy_func
