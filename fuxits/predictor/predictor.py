import fuxits.losses as losses
from fuxits.utils import set_color, parser_yaml, color_dict, print_logger, xavier_normal_initialization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger 
from fuxits.data import TSDataset
import os, torch
from torch import optim, nn
from fuxits.metrics.predmetric import get_pred_metrics
class Predictor(LightningModule):
    def __init__(self, config):
        super(Predictor, self).__init__()
        if config is None:
            config = parser_yaml(os.path.join(os.path.dirname(__file__), "predictor.yaml"))
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.model_config = config['model']
        self.missing_value = config['missing_value'] if 'missing_value' in config else None
        if self.train_config['seed'] is not None:
            seed_everything(self.train_config['seed'], workers=True)
        self.loss = self._get_loss()
        self.loss = losses.MaskedLoss(self.loss, self.missing_value)
        self.save_hyperparameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def fit(self, train_data:TSDataset, val_data:TSDataset=None, run_mode='detail'):
        self.run_mode = run_mode
        self.val_check = val_data is not None and self.eval_config['val_metrics'] is not None
        if run_mode == 'tune' and "NNI_OUTPUT_DIR" in os.environ: 
            save_dir = os.environ["NNI_OUTPUT_DIR"] #for parameter tunning
        else:
            save_dir = os.getcwd()
        print_logger.info('save_dir:' + save_dir)
        #refresh_rate = 0 if run_mode in ['light', 'tune'] else 1
        logger = TensorBoardLogger(save_dir=save_dir, name="tensorboard")
        self.reset_parameters()
        train_loader = train_data.loader(batch_size=self.train_config['batch_size'], \
            shuffle=False, num_workers=self.train_config['num_workers']) ## to do, replace shuffle with True
        if val_data:
            val_loader = val_data.loader(batch_size=self.eval_config['batch_size'],\
            shuffle=False, num_workers=self.eval_config['num_workers'])
        else:
            val_loader = None
        trainer = Trainer(devices=self.train_config['devices'], 
                            max_epochs=self.train_config['epochs'], 
                            num_sanity_val_steps=0,
                            logger=logger,
                            accelerator=self.train_config['accelerator'])
        trainer.fit(self, train_loader, val_loader)


    def evaluate(self, test_data:TSDataset, verbose=True, **kwargs):
        test_loader = test_data.loader(batch_size=self.eval_config['batch_size'], num_workers=self.eval_config['num_workers'])
        output = self.trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=False)[0] ## indicate the first dataset (the only one)
        if verbose:
            print_logger.info(color_dict(output, self.run_mode=='tune'))
        return output

    def predict(self, test_data:TSDataset):
        test_loader = test_data.loader(batch_size=self.eval_config['batch_size'], num_workers=self.eval_config['num_workers'])
        output = self.trainer.test(dataloaders=test_loader, ckpt_path='best', verbose=False)[0]
        return output

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def test_step(self, batch, batch_idx):
        return self._test_step(batch, get_pred_metrics(self.eval_config['test_metrics']))

    def _test_step(self, batch, metrics):
        x, y = batch
        y_hat = self.forward(x)
        if self.missing_value is not None:
            return [func(y_hat, y, self.missing_value) for _, func in metrics], losses.ismask(y, self.missing_value).sum() 
        else:
            return [func(y_hat, y) for _, func in metrics], y.numel()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._test_step(batch, get_pred_metrics(self.eval_config['val_metrics']))

    def configure_callbacks(self):
        if self.val_check:
            eval_metric = self.eval_config['val_metrics']
            self.val_metric = next(iter(eval_metric)) if isinstance(eval_metric, list)  else eval_metric
            early_stopping = EarlyStopping(self.val_metric, verbose=True, patience=self.eval_config['early_stop_patience'], \
                mode = self.eval_config['early_stop_mode'])
            ckp_callback = ModelCheckpoint(monitor=self.val_metric, save_top_k=1, \
                mode = self.eval_config['early_stop_mode'], save_last=True)
            return [ckp_callback, early_stopping]
    
    def _get_loss(self):
        pass

    def validation_epoch_end(self, outputs):
        self._test_epoch_end(outputs, True)

    def test_epoch_end(self, outputs):
        self._test_epoch_end(outputs, False)
    
    def _test_epoch_end(self, outputs, is_valid):
        metric_name = self.eval_config['val_metrics' if is_valid else 'test_metrics']
        if not isinstance(metric_name, list):
            metric_name = [metric_name]
        metric, numel = zip(*outputs)
        metric = torch.tensor(metric)
        numel = torch.tensor(numel)
        metric = (metric * numel.view(-1, 1)).sum(0) / numel.sum()
        self.log_dict(dict(zip(metric_name, metric)), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = self.get_optimizer(params)
        scheduler = self.get_scheduler(optimizer)
        m = self.val_metric if self.val_check else 'train_loss'
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': m,
                    'interval': 'epoch',
                    'frequency': 1,
                    'strict': False
                }
            }
        else:
            return optimizer

    def get_optimizer(self, params):
        '''@nni.variable(nni.choice(0.1, 0.05, 0.01, 0.005, 0.001), name=learning_rate)'''
        learning_rate = self.train_config['learning_rate']
        '''@nni.variable(nni.choice(0.1, 0.01, 0.001, 0), name=decay)'''
        decay = self.train_config['weight_decay']
        if self.train_config['learner'].lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
        elif self.train_config['learner'].lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=decay)
        elif self.train_config['learner'].lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=decay)
        elif self.train_config['learner'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay)
        elif self.train_config['learner'].lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            #if self.weight_decay > 0:
            #    self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def get_scheduler(self, optimizer):
        if self.train_config['scheduler'] is not None:
            if self.train_config['scheduler'].lower() == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            elif self.train_config['scheduler'].lower() == 'onplateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            else:
                scheduler = None
        else:
            scheduler = None
        return scheduler
