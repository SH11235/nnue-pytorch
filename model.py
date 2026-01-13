import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import math

# アーキテクチャプリセット定義
# 形式: (L1, L2, L3)
# 命名規則: halfkp_{L1}x2-{L2}-{L3}
ARCH_PRESETS = {
    'halfkp_256x2-32-32': (256, 32, 32),   # やねうら王標準NNUE（水匠5、Háo等）
    'halfkp_1024x2-8-32': (1024, 8, 32),   # tanuki- Lí (WCSC33)
    'halfkp_512x2-8-96': (512, 8, 96),     # カスタム設定
}

# デフォルトアーキテクチャ
DEFAULT_ARCH = 'halfkp_256x2-32-32'

def get_arch_sizes(arch_name=None, l1=None, l2=None, l3=None):
    """
    アーキテクチャサイズを取得する。
    個別指定（l1, l2, l3）がある場合はそちらを優先。
    """
    if arch_name and arch_name in ARCH_PRESETS:
        base_l1, base_l2, base_l3 = ARCH_PRESETS[arch_name]
    else:
        base_l1, base_l2, base_l3 = ARCH_PRESETS[DEFAULT_ARCH]

    # 個別指定で上書き
    final_l1 = l1 if l1 is not None else base_l1
    final_l2 = l2 if l2 is not None else base_l2
    final_l3 = l3 if l3 is not None else base_l3

    return final_l1, final_l2, final_l3

def list_arch_presets():
    """利用可能なプリセット一覧を返す"""
    return list(ARCH_PRESETS.keys())

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(
      self, feature_set, lambda_=[1.0], lr=[1.0],
      label_smoothing_eps=0.0, num_batches_warmup=10000, newbob_decay=0.5,
      num_epochs_to_adjust_lr=500, score_scaling=361, min_newbob_scale=1e-5,
      momentum=0.0, ply_begin_threshold=100.0, ply_end_threshold=120.0,
      arch=None, l1_size=None, l2_size=None, l3_size=None,
      lr_milestones=None, lr_gamma=0.2):
    super(NNUE, self).__init__()

    # アーキテクチャサイズを決定
    self.L1, self.L2, self.L3 = get_arch_sizes(arch, l1_size, l2_size, l3_size)
    self.arch_name = arch or DEFAULT_ARCH

    self.input = nn.Linear(feature_set.num_features, self.L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * self.L1, self.L2)
    self.l2 = nn.Linear(self.L2, self.L3)
    self.output = nn.Linear(self.L3, 1)
    self.lambda_ = lambda_
    self.lr = lr
    self.label_smoothing_eps = label_smoothing_eps
    self.num_batches_warmup = num_batches_warmup
    self.newbob_scale = 1.0
    self.newbob_decay = newbob_decay
    self.best_loss = 1e10
    self.num_epochs_to_adjust_lr = num_epochs_to_adjust_lr
    self.latest_loss_sum = 0.0
    self.latest_loss_count = 0
    self.score_scaling = score_scaling
    # Warmupを開始するステップ数
    self.warmup_start_global_step = 0
    self.min_newbob_scale = min_newbob_scale
    self.parameter_index = 0
    self.momentum = momentum
    self.ply_begin_threshold = ply_begin_threshold
    self.ply_end_threshold = ply_end_threshold
    # LR schedule (MultiStepLR)
    self.lr_milestones = lr_milestones if lr_milestones is not None else []
    self.lr_gamma = lr_gamma

    self._zero_virtual_feature_weights()

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type):
    us, them, white, black, outcome, score, ply = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    nnue2score = 600
    scaling = self.score_scaling

    q = self(us, them, white, black) * nnue2score / scaling
    t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
    p = (score / scaling).sigmoid()

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
    if self.lambda_[self.parameter_index] >= 0.0:
      lambda_ = self.lambda_[self.parameter_index]
    else:
      lambda_ = (self.ply_end_threshold - ply) / (self.ply_end_threshold - self.ply_begin_threshold)
      lambda_ = torch.clamp(lambda_ , 0.0, 1.0)
    result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    loss = result.mean() - entropy.mean()
    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'val_loss')
  
  def on_validation_epoch_end(self):
    # Lightning 2.x: use on_validation_epoch_end without outputs parameter
    # Collect loss from logged metrics
    val_loss = self.trainer.callback_metrics.get('val_loss')
    if val_loss is not None:
      self.latest_loss_sum += float(val_loss)
      self.latest_loss_count += 1

    if self.newbob_decay != 1.0 and self.current_epoch > 0 and self.current_epoch % self.num_epochs_to_adjust_lr == 0:
      latest_loss = self.latest_loss_sum / self.latest_loss_count if self.latest_loss_count > 0 else 1e10
      self.latest_loss_sum = 0.0
      self.latest_loss_count = 0
      if latest_loss < self.best_loss:
        self.print(f"{self.current_epoch=}, {latest_loss=} < {self.best_loss=}, accepted, {self.newbob_scale=}")
        sys.stdout.flush()
        self.best_loss = latest_loss
      else:
        self.newbob_scale *= self.newbob_decay
        self.print(f"{self.current_epoch=}, {latest_loss=} >= {self.best_loss=}, rejected, {self.newbob_scale=}")
        sys.stdout.flush()

    if self.newbob_scale < self.min_newbob_scale:
      self.parameter_index += 1
      if self.parameter_index < len(self.lr):
        self.best_loss = 1e10
        self.newbob_scale = 1.0
      else:
        self.trainer.should_stop = True
        self.print(f"{self.current_epoch=}, early stopping")

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  # learning rate warm-up (Lightning 2.x compatible)
  def on_before_optimizer_step(self, optimizer):
    # MultiStepLRを使用している場合もwarmupを適用
    if self.lr_milestones:
      warmup_scale = 1.0
      if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
        warmup_scale = min(1.0, float(self.trainer.global_step - self.warmup_start_global_step + 1) / self.num_batches_warmup)

      # スケジューラが設定したLRにwarmup係数を掛ける
      scheduler = self.lr_schedulers()
      if scheduler is not None:
        scheduled_lr = scheduler.get_last_lr()[0]
      else:
        scheduled_lr = self.lr[0]

      for pg in optimizer.param_groups:
        pg["lr"] = scheduled_lr * warmup_scale
        self.log("lr", pg["lr"])
      return

    # manually warm up lr without a scheduler
    if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
      warmup_scale = min(1.0, float(self.trainer.global_step - self.warmup_start_global_step + 1) / self.num_batches_warmup)
    else:
      warmup_scale = 1.0
    for pg in optimizer.param_groups:
      pg["lr"] = self.lr[self.parameter_index] * warmup_scale * self.newbob_scale
      self.log("lr", pg["lr"])

  # weight clipping after optimizer step (Lightning 2.x compatible)
  def on_train_batch_end(self, outputs, batch, batch_idx):
    # clip parameters after weight update
    for child in self.children():
      if not isinstance(child, nn.Linear):
        continue

      if child == self.input:
        continue

      # FC layers are stored as int8 weights, and int32 biases
      kWeightScaleBits = 6
      kActivationScale = 127.0
      if child != self.output:
        kBiasScale = (1 << kWeightScaleBits) * kActivationScale # = 8128
      else:
        kBiasScale = 9600.0 # kPonanzaConstant * FV_SCALE = 600 * 16 = 9600
      kWeightScale = kBiasScale / kActivationScale # = 64.0 for normal layers
      kMaxWeight = 127.0 / kWeightScale # roughly 2.0
      child.weight.data.clamp_(-kMaxWeight, kMaxWeight)

  def configure_optimizers(self):
    # v12: v10設定に回帰（一律weight_decay=1e-4）
    # v11のparam group分離はデータ問題（move16非合法）と切り分けるため一旦戻す
    optimizer = torch.optim.SGD(
        self.parameters(),
        lr=self.lr[0],
        momentum=self.momentum,
        weight_decay=1e-4
    )

    # LR schedule: MultiStepLRを使用（lr_milestonesが指定されている場合）
    if self.lr_milestones:
      scheduler = torch.optim.lr_scheduler.MultiStepLR(
          optimizer,
          milestones=self.lr_milestones,
          gamma=self.lr_gamma
      )
      print(f"Using MultiStepLR: milestones={self.lr_milestones}, gamma={self.lr_gamma}")
      return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    return optimizer

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
