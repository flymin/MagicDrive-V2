import sys
import types
import importlib


class NoneModule(types.ModuleType):
    def __getattr__(self, name):
        return None


class NoneModuleLoader:
    def __init__(self, prefix):
        self.prefix = prefix

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(self.prefix):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return NoneModule(spec.name)

    def exec_module(self, module):
        pass


# Define the module prefix to intercept
prefixes = [
    'colossalai.quantization',
    'mmcv._ext',  # we use 1.7.2, _ext is not compatible with latest CANN
]

# Create and insert the custom loader
for prefix in prefixes:
    loader = NoneModuleLoader(prefix)
    sys.meta_path.insert(0, loader)
