from fuxits.utils.utils import set_color, parser_yaml, color_dict, print_logger, xavier_normal_initialization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger 
from fuxits.data.dataset import TSDataset
import os, torch
class Predictor(LightningModule):
    def __init__(self, config):
        super(Predictor, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = parser_yaml(os.path.join(os.path.dirname(__file__), "predictor.yaml"))
        if self.config['seed'] is not None:
            seed_everything(self.config['seed'], workers=True)
        self.config = config
        self.loss_fn = self._get_loss_func()
        self.save_hyperparameters()
        
    def reset_parameters(self):
        self.apply(xavier_normal_initialization)

    def fit(self, train_data:TSDataset, val_data:TSDataset=None, run_mode='detail'):
        self.run_mode = run_mode
        self.val_check = val_data is not None and self.config['val_metrics'] is not None
        if run_mode == 'tune' and "NNI_OUTPUT_DIR" in os.environ: 
            save_dir = os.environ["NNI_OUTPUT_DIR"] #for parameter tunning
        else:
            save_dir = os.getcwd()
        print_logger.info('save_dir:' + save_dir)
        refresh_rate = 0 if run_mode in ['light', 'tune'] else 1
        logger = TensorBoardLogger(save_dir=save_dir, name="tensorboard")
        self.reset_parameters()
        train_loader = train_data.loader(batch_size=self.config['batch_size'], \
            shuffle=True, num_workers=self.config['num_workers'])
        if val_data:
            val_loader = val_data.loader(batch_size=self.config['eval_batch_size'],\
            shuffle=False, num_workers=self.config['num_workers'])
        else:
            val_loader = None
        trainer = Trainer(gpus=self.config['gpu'], 
                            max_epochs=self.config['epochs'], 
                            num_sanity_val_steps=0,
                            progress_bar_refresh_rate=refresh_rate,
                            logger=logger,
                            accelerator="dp")
        trainer.fit(self, train_loader, val_loader)

    def evaluate(self, test_data:TSDataset, verbose=True, **kwargs):
        test_loader = test_data.loader(batch_size=self.config['eval_batch_size'], num_workers=self.config['num_workers'])
        output = self.trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=False)[0] ## indicate the first dataset (the only one)
        if verbose:
            print_logger.info(color_dict(output, self.run_mode=='tune'))
        return output

    def predict(self, test_data:TSDataset):
        test_loader = test_data.loader(batch_size=self.config['eval_batch_size'], num_workers=self.config['num_workers'])
        output = self.trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=False)[0]
        return output

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def test_step(self, batch, batch_idx):
        return self._test_step(batch, self.config['test_metrics'])

    def _test_step(self, batch, metrics):
        y_pred = self.forward(batch)
        y_true = batch['y']
        return [func(y_pred, y_true) for _, func in metrics]
        
    def training_step(self, batch, batch_idx):
        y_ = self.forward(batch)
        y = batch['y']
        loss = self.loss_fn(y, y_)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._test_step(batch, self.config['val_metrics'])

    def configure_callbacks(self):
        early_stopping = EarlyStopping(self.val_metric, verbose=True, patience=10, \
            mode=self.config['early_stop_mode'])
        ckp_callback = ModelCheckpoint(monitor=self.val_metric, save_top_k=1, \
            mode=self.config['early_stop_mode'], save_last=True)
        return [ckp_callback, early_stopping]
    
    def _get_loss_func(self):
        pass

    def validation_epoch_end(self, outputs):
        self._test_epoch_end(outputs, True)

    def test_epoch_end(self, outputs):
        self._test_epoch_end(outputs, False)
    
    def _test_epoch_end(self, outputs, is_valid):
        metric_name = 'val_metrics' if is_valid else 'test_metrics'
        metric_name = self.config[metric_name] 
        if not isinstance(self.config[metric_name], list):
            metric_name = [metric_name]
        metric, bs = zip(*outputs)
        metric = torch.tensor(metric)
        bs = torch.tensor(bs)
        metric = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        self.log_dict(dict(zip(metric_name, metric)), on_step=False, on_epoch=True)