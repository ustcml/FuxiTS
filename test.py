import pkgutil, importlib
import fuxits
from fuxits.utils.timeparse import stepparse
# def find_module(package, mod_name):
#     if isinstance(package, str):
#         package = importlib.import_module(package)
#     for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
#         print(name)
#         # if ispkg:
#         #     fullname = '.'.join([package.__name__, name, mod_name])
#         #     if importlib.util.find_spec(fullname, __name__):
#         #         return importlib.import_module(fullname)
#         #     else:
#         #         return find_module(importlib.import_module('.'.join([package.__name__, name])), mod_name)
#     return None



# module = find_module('fuxits', 'astgcn')



print(stepparse(['6d','12w', '3h']))