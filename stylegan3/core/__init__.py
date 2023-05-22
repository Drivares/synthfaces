import sys
from . import dnnlib
sys.modules['dnnlib'] = dnnlib
from . import gen_utils
sys.modules['gen_utils'] = gen_utils
from . import gui_utils
sys.modules['gui_utils'] = gui_utils
from . import torch_utils
sys.modules['torch_utils'] = torch_utils