import os, argparse
from fuxits.utils import get_model, print_logger, color_dict_normal, set_color
from fuxits.data.dataset import TSDataset
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='ASTGCN', help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='METR_LA', help='dataset name')
    parser.add_argument('--mode', choices=['tune', 'light', 'detail'], default='light', help='flag indiates model tuning')
    args, commond_line_args = parser.parse_known_args()
    model_class, model_conf = get_model(args.model)
    args_ = parser.parse_args(commond_line_args)
    for opt in commond_line_args:
        key = opt.split('=')[0].strip('-')
        value = getattr(args_, key)
        model_conf[key] = value
    data = TSDataset(args.dataset)
    datasets = data.build(model_conf['split_ratio'], model_conf['model']['hist_steps'], model_conf['model']['pred_steps'])
    model = model_class(model_conf, **data.datafeatures)
    model.fit(*datasets[:2], run_mode=args.mode)
    model.evaluate(datasets[-1])
    

