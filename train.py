import argparse
import model as M
import nnue_dataset
import nnue_bin_dataset
import pytorch_lightning as pl
import features
import os
import torch
import pytorch_lightning.callbacks
import typing
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

def data_loader_cc(train_filenames, val_filenames, feature_set, num_workers, batch_size, filtered, random_fen_skipping, main_device, epoch_size):
  # Epoch and validation sizes are arbitrary
  val_size = 1000000
  features_name = feature_set.name
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filenames, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filenames, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def data_loader_py(train_filename, val_filename, feature_set, batch_size, main_device):
  train = DataLoader(nnue_bin_dataset.NNUEBinData(train_filename, feature_set), batch_size=batch_size, shuffle=True, num_workers=4)
  val = DataLoader(nnue_bin_dataset.NNUEBinData(val_filename, feature_set), batch_size=32)
  return train, val


class NetworkSaveCheckpoint(pytorch_lightning.callbacks.Checkpoint):
  def __init__(
      self,
      every_n_epochs: int,
      log_dir: str,
  ):
    self.every_n_epochs = every_n_epochs
    self.log_dir = log_dir

  def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    if trainer.current_epoch == 0 or trainer.current_epoch % self.every_n_epochs != 0:
      return

    ckpt_file_path = os.path.join(self.log_dir, f'{trainer.current_epoch}.ckpt')
    trainer.save_checkpoint(ckpt_file_path)


class FeatureCollapseCallback(pytorch_lightning.callbacks.Callback):
  """Feature Collapseを検出して学習を停止するコールバック"""

  def __init__(self, min_active_rate: float = 0.5, check_every_n_epochs: int = 10):
    self.min_active_rate = min_active_rate  # 最小活性化率（デフォルト50%）
    self.check_every_n_epochs = check_every_n_epochs
    self.best_active_rate = 0.0
    self.best_epoch = 0

  def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    if trainer.current_epoch == 0:
      return
    if trainer.current_epoch % self.check_every_n_epochs != 0:
      return

    # input.biasを取得
    bias = pl_module.input.bias.detach().cpu().numpy()

    # 活性化率を計算（bias > -1/127 の割合）
    threshold = -1.0 / 127.0
    active_rate = (bias > threshold).sum() / len(bias)

    # ベスト記録を更新
    if active_rate > self.best_active_rate:
      self.best_active_rate = active_rate
      self.best_epoch = trainer.current_epoch

    print(f"\n[Feature Collapse Check] Epoch {trainer.current_epoch}")
    print(f"  Bias Mean: {bias.mean():.4f}, Range: [{bias.min():.4f}, {bias.max():.4f}]")
    print(f"  Active Rate: {active_rate*100:.1f}% (threshold: {self.min_active_rate*100:.0f}%)")
    print(f"  Best: {self.best_active_rate*100:.1f}% at epoch {self.best_epoch}")

    if active_rate < self.min_active_rate:
      print(f"\n⚠️ Feature Collapse検出! 活性化率 {active_rate*100:.1f}% < {self.min_active_rate*100:.0f}%")
      print(f"学習を停止します。Best checkpointはepoch {self.best_epoch}です。")
      trainer.should_stop = True


def main():
  parser = argparse.ArgumentParser(description="Trains the network.")
  parser.add_argument("train", help="Training data (.bin). Multiple files can be specified with comma-separated paths (e.g., 'file1.bin,file2.bin')")
  parser.add_argument("val", help="Validation data (.bin). Multiple files can be specified with comma-separated paths")
  # Lightning 2.x: Trainer args are added manually instead of add_argparse_args
  parser.add_argument("--accelerator", default="auto", help="Accelerator to use (auto, cpu, gpu, etc.)")
  parser.add_argument("--devices", default="auto", help="Number of devices to use")
  parser.add_argument("--max-epochs", default=-1, type=int, dest='max_epochs', help="Maximum number of epochs")
  parser.add_argument("--default-root-dir", default=None, dest='default_root_dir', help="Default root directory for logs")
  parser.add_argument("--py-data", action="store_true", help="Use python data loader (default=False)")
  parser.add_argument("--lambda", default=[1.0], nargs='+', type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
  parser.add_argument("--lr", default=[1.0], nargs='+', type=float, dest='lr', help="Initial learning rate.")
  parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for binpack.")
  parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
  parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
  parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping', help="If enabled positions that are bad training targets will be skipped during loading. Default: False")
  parser.add_argument("--random-fen-skipping", default=0, type=int, dest='random_fen_skipping', help="skip fens randomly on average random_fen_skipping before using one.")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initializes training using the weights from the given .pt model")
  parser.add_argument("--network-save-period", type=int, default=1000000000, dest='network_save_period', help="Number of epochs between network snapshots. None to disable.")
  parser.add_argument("--label-smoothing-eps", default=0.0, type=float, dest='label_smoothing_eps', help="Label smoothing eps.")
  parser.add_argument("--num-batches-warmup", default=10000, type=int, dest='num_batches_warmup', help="Number of batches for warm-up.")
  parser.add_argument("--newbob-decay", default=0.5, type=float, dest='newbob_decay', help="Newbob decay.")
  parser.add_argument("--epoch-size", default=10000000, type=int, dest='epoch_size', help="epoch size.")
  parser.add_argument("--num-epochs-to-adjust-lr", default=50, type=int, dest='num_epochs_to_adjust_lr', help="Number of epochs to adjust learning rate.")
  parser.add_argument("--score-scaling", default=361, type=float, dest='score_scaling', help="Score scaling.")
  parser.add_argument("--min-newbob-scale", default=1e-5, type=float, dest='min_newbob_scale', help="Minimum learning rate to stop the training.")
  parser.add_argument("--momentum", default=0.0, type=float, dest='momentum', help="Momentum.")
  parser.add_argument("--lr-milestones", default=[], nargs='*', type=int, dest='lr_milestones', help="Epochs at which to decay LR (e.g., --lr-milestones 15 25).")
  parser.add_argument("--lr-gamma", default=0.2, type=float, dest='lr_gamma', help="LR decay factor at each milestone (default: 0.2).")
  parser.add_argument("--ply-begin-threshold", default=100.0, type=float, dest='ply_begin_threshold', help="Ply at which lambda begins to decay.")
  parser.add_argument("--ply-end-threshold", default=120.0, type=float, dest='ply_end_threshold', help="Ply at which lambda ends to decay.")
  parser.add_argument("--min-active-rate", default=0.5, type=float, dest='min_active_rate', help="Minimum active neuron rate before stopping (Feature Collapse detection). Set to 0 to disable.")

  # アーキテクチャ設定
  arch_choices = M.list_arch_presets()
  parser.add_argument("--arch", default=None, choices=arch_choices, dest='arch',
                      help=f"Architecture preset. Available: {', '.join(arch_choices)}. Default: {M.DEFAULT_ARCH}")
  parser.add_argument("--l1", default=None, type=int, dest='l1_size',
                      help="L1 layer size (Feature Transformer output). Overrides --arch if specified.")
  parser.add_argument("--l2", default=None, type=int, dest='l2_size',
                      help="L2 layer size. Overrides --arch if specified.")
  parser.add_argument("--l3", default=None, type=int, dest='l3_size',
                      help="L3 layer size. Overrides --arch if specified.")

  features.add_argparse_args(parser)
  args = parser.parse_args()

  # Parse comma-separated file lists
  train_files = [f.strip() for f in args.train.split(',')]
  val_files = [f.strip() for f in args.val.split(',')]

  # Check all files exist
  for f in train_files:
    if not os.path.exists(f):
      raise Exception('{0} does not exist'.format(f))
  for f in val_files:
    if not os.path.exists(f):
      raise Exception('{0} does not exist'.format(f))

  feature_set = features.get_feature_set_from_name(args.features)

  # アーキテクチャサイズを決定・表示
  arch_l1, arch_l2, arch_l3 = M.get_arch_sizes(args.arch, args.l1_size, args.l2_size, args.l3_size)
  arch_name = args.arch or M.DEFAULT_ARCH
  print(f"Architecture: {arch_name} (L1={arch_l1}, L2={arch_l2}, L3={arch_l3})")

  if not args.resume_from_model:
    nnue = M.NNUE(
      feature_set=feature_set, lambda_=args.lambda_,
      lr=args.lr, label_smoothing_eps=args.label_smoothing_eps,
      num_batches_warmup=args.num_batches_warmup,
      newbob_decay=args.newbob_decay,
      num_epochs_to_adjust_lr=args.num_epochs_to_adjust_lr,
      score_scaling=args.score_scaling,
      min_newbob_scale=args.min_newbob_scale, momentum=args.momentum,
      ply_begin_threshold=args.ply_begin_threshold, ply_end_threshold=args.ply_end_threshold,
      arch=args.arch, l1_size=args.l1_size, l2_size=args.l2_size, l3_size=args.l3_size,
      lr_milestones=args.lr_milestones, lr_gamma=args.lr_gamma)
  else:
    # モデルファイルの形式を判定
    checkpoint = torch.load(args.resume_from_model, map_location='cpu')
    is_lightning_ckpt = 'pytorch-lightning_version' in checkpoint
    is_converted_model = 'state_dict' in checkpoint and 'architecture' in checkpoint

    if is_lightning_ckpt:
      # PyTorch Lightningチェックポイント形式
      print(f'Loading Lightning checkpoint: {args.resume_from_model}')
      nnue = M.NNUE.load_from_checkpoint(
        args.resume_from_model, feature_set=feature_set,
        arch=args.arch, l1_size=args.l1_size, l2_size=args.l2_size, l3_size=args.l3_size)
    elif is_converted_model:
      # 変換済みモデル形式（state_dict + architecture）
      print(f'Loading converted model: {args.resume_from_model}')
      print(f'  Original architecture: {checkpoint.get("architecture", "unknown")}')
      nnue = M.NNUE(
        feature_set=feature_set, lambda_=args.lambda_,
        lr=args.lr, label_smoothing_eps=args.label_smoothing_eps,
        num_batches_warmup=args.num_batches_warmup,
        newbob_decay=args.newbob_decay,
        num_epochs_to_adjust_lr=args.num_epochs_to_adjust_lr,
        score_scaling=args.score_scaling,
        min_newbob_scale=args.min_newbob_scale, momentum=args.momentum,
        ply_begin_threshold=args.ply_begin_threshold, ply_end_threshold=args.ply_end_threshold,
        arch=args.arch, l1_size=args.l1_size, l2_size=args.l2_size, l3_size=args.l3_size,
        lr_milestones=args.lr_milestones, lr_gamma=args.lr_gamma)
      nnue.load_state_dict(checkpoint['state_dict'])
    else:
      raise ValueError(f'Unknown checkpoint format: {args.resume_from_model}')

    nnue.set_feature_set(feature_set)
    nnue.lambda_ = args.lambda_
    # we can set the following here just like that because when resuming
    # from .pt the optimizer is only created after the training is started
    nnue.lr = args.lr
    nnue.label_smoothing_eps=args.label_smoothing_eps
    nnue.num_batches_warmup=args.num_batches_warmup
    nnue.newbob_decay=args.newbob_decay
    nnue.num_epochs_to_adjust_lr=args.num_epochs_to_adjust_lr
    nnue.score_scaling=args.score_scaling
    nnue.min_newbob_scale=args.min_newbob_scale
    nnue.momentum=args.momentum

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))

  print("Training with {} validating with {}".format(train_files, val_files))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  batch_size = args.batch_size
  if batch_size <= 0:
    # Lightning 2.x: use accelerator instead of gpus
    use_gpu = args.accelerator in ("gpu", "cuda", "auto") and torch.cuda.is_available()
    batch_size = 8192 if use_gpu else 128
  print('Using batch size {}'.format(batch_size))

  print('Smart fen skipping: {}'.format(args.smart_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  checkpoint_callback = NetworkSaveCheckpoint(every_n_epochs=args.network_save_period, log_dir=tb_logger.log_dir)

  # コールバックリストを構築
  callbacks = [checkpoint_callback]
  if args.min_active_rate > 0:
    collapse_callback = FeatureCollapseCallback(
      min_active_rate=args.min_active_rate,
      check_every_n_epochs=args.network_save_period
    )
    callbacks.append(collapse_callback)
    print(f'Feature Collapse detection enabled: min_active_rate={args.min_active_rate*100:.0f}%')

  # Lightning 2.x: use Trainer() directly instead of from_argparse_args
  trainer = pl.Trainer(
    accelerator=args.accelerator,
    devices=args.devices,
    max_epochs=args.max_epochs if args.max_epochs > 0 else None,
    default_root_dir=args.default_root_dir,
    callbacks=callbacks,
    logger=tb_logger,
    num_sanity_val_steps=0,  # Skip sanity check to avoid issues
  )

  main_device = 'cuda:0'

  if args.py_data:
    print('Using python data loader')
    # Python data loader only supports single file
    train, val = data_loader_py(train_files[0], val_files[0], feature_set, batch_size, main_device)
  else:
    print('Using c++ data loader')
    train, val = data_loader_cc(train_files, val_files, feature_set, args.num_workers, batch_size, args.smart_fen_skipping, args.random_fen_skipping, main_device, args.epoch_size)

  trainer.fit(nnue, train, val)

  print(f'tb_logger.log_dir={tb_logger.log_dir}')
  ckpt_file_path = os.path.join(tb_logger.log_dir, 'final.ckpt')
  trainer.save_checkpoint(ckpt_file_path)


if __name__ == '__main__':
  main()
