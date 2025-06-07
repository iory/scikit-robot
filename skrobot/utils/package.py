def is_package_installed(package_name):
    try:
        from importlib.metadata import distribution
        from importlib.metadata import PackageNotFoundError
        try:
            distribution(package_name)
            return True
        except PackageNotFoundError:
            return False
    except ImportError:
        import imp
        try:
            imp.find_module(package_name)
            return True
        except ImportError:
            return False
