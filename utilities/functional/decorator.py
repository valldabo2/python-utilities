import functools


def decorator_bye(func):
    @functools.wraps(func)
    def wrapper(name):
        res = func(name)
        print(f"bye:{name}")
        return res
    return wrapper

@decorator_bye
def hi(name):
    print(f"hi:{name}")


hi("Someone")
