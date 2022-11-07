def dummy_func(*args, **kwargs):
    message = "cvxopt is not installed. please install cvxopt by 'pip install cvxopt' if wheel is avilable for your platform"
    raise ModuleNotFoundError(message)

try:
    from skrobot.optimizers import cvxopt_solver  # NOQA
    from skrobot.optimizers import quadprog_solver  # NOQA
except ImportError:
    cvxopt_solver = dummy_func
    quadprog_solver = dummy_func
