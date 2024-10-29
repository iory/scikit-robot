def is_package_installed(package_name):
    try:
        import pkg_resources
        try:
            pkg_resources.get_distribution(package_name)
            return True
        except pkg_resources.DistributionNotFound:
            return False
    except ImportError:
        import imp
        try:
            imp.find_module(package_name)
            return True
        except ImportError:
            return False
