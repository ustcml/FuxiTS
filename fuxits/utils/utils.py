import yaml, re, importlib, os, logging, pkgutil
print_logger = logging.getLogger("pytorch_lightning")
print_logger.setLevel(logging.INFO)
def set_color(log, color, highlight=True, keep=False):
    if keep:
        return log
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'


def parser_yaml(config_path):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        ret = yaml.load(f.read(), Loader=loader)
    return ret

def color_dict(dict_, keep=True):
    key_color = 'blue'
    val_color = 'yellow'
    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) +'=' + set_color(v_f, val_color, keep=keep)) % (k, v)
        return info

    des = 4
    if 'epoch' in dict_:
        start = set_color('Training: ', 'green', keep=keep)
        start += color_kv('Epoch', dict_['epoch'], '%s', '%3d')
    else:
        start = set_color('Testing: ', 'green', keep=keep)
    info = ' '.join([ color_kv(k, v, '%s', '%.'+str(des)+'f') for k, v in dict_.items() if k != 'epoch'])   
    return start + ' ['+ info + ']'

def color_dict_normal(dict_, keep=True):
    key_color = 'blue'
    val_color = 'yellow'
    def color_kv(k, v, k_f, v_f):
        info = (set_color(k_f, key_color, keep=keep) +'=' + set_color(v_f, val_color, keep=keep)) % (k, v)
        return info
    info = '\n'.join([ color_kv(k, str(v), '%s', '%s') for k, v in dict_.items()])   
    return info


def get_model(model_name):
    import fuxits
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = ['predictor', 'detector', 'explainer', 'intervener']
    for id, m in enumerate(model_submodule):
        module = importlib.import_module('fuxits.' + m)
        model_module = find_model(module, model_name.lower())
        if model_module is not None:
            break
    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    else:
        model_module = importlib.import_module(model_module)
    model_class = getattr(model_module, model_name)
    dir = os.path.dirname(model_module.__file__)
    conf = dict()
    fname = os.path.join(os.path.dirname(dir), 'config/{}.yml'.format(model_submodule[id]))
    conf.update(parser_yaml(fname))
    fname = os.path.join(dir, 'config', model_name+'.yml')
    if os.path.isfile(fname):
        conf.update(parser_yaml(fname))
    return model_class, conf

def find_model(package, model_name):
    if isinstance(package, str):
        package = importlib.import_module(package)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg and name == model_name:
            return package.__name__ + '.' + name
        if ispkg:
            return find_model(package.__name__+'.'+ name, model_name)
    return None



def xavier_normal_initialization(module):
    import torch.nn as nn
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
        nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)



def xavier_uniform_initialization(module):
    import torch.nn as nn
    r""" using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
