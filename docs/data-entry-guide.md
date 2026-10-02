# Data entry guide

How papers are turned into Heedless Backbones records. It is written for the
extraction model as much as for people: it states the conventions already followed in
the database so the model can apply them, and use judgment where they run out.

From a review of the stored data (54 families), `todo.txt` and the original `llm_gen`
prompt, plus the maintainer's answers. The extraction prompt includes everything
between the markers below; update it here.

<!-- prompt:start -->

## Judgment

- When these rules do not cover a case, make the call a careful maintainer would make
  from the rules' intent, explain it in the record's note, and continue. A judgment
  call is not a reason for review.
- Send a paper to review only for missing or contradictory data: a required value the
  paper does not give, figures that disagree within the paper, or a value that cannot
  be read reliably (e.g. an unreadable table).
- Every value still needs a quotation from the paper. Conventions below (dataset
  sizes, schedule names) may be applied without an extra citation.

## Sources

- The paper is the primary source. The official repository (README, model zoo tables,
  release notes) and project page linked from the paper may supply values for the
  paper's own models that the paper does not give (e.g. throughput), cited by URL.
- Also include models of the family that only the repository reports, such as
  checkpoints released after the paper (e.g. Swin-T and Swin-S pretrained on
  ImageNet-22k), when the repository gives the same kind of results as for the paper's
  models (e.g. ImageNet-1k accuracy with parameters and FLOPs). Record their paper link
  as the repository URL.
- When the paper and its repository disagree, use the paper's value and note the
  repository's.

## Conventions

Standard values to use when a paper follows a common setup without restating it. Use
them without a citation (as `source: "convention: <name>"` for a derivation input) and
say in the note that the convention was applied.

- **Dataset sizes** for iteration-to-epoch conversions: ADE20K = 20,000 images,
  COCO train2017 = 117,000, Cityscapes = 3,000.
- **Batch size 16** for downstream training that follows a standard mmsegmentation or
  mmdetection setup, or another backbone's official setup (ConvNeXt, Swin, PVT), when
  the paper does not state it. So ADE20K UPerNet 160k iterations = 128 epochs, and
  Semantic/Panoptic FPN 80k iterations = 64 epochs.
- **Named detection schedules**: 1x = 12 epochs, 2x = 24, 3x = 36, 6x = 72.
- **Crop size** 512 for ADE20K and 1024 for Cityscapes when the setup is standard and
  unstated (640 only when the paper says so).
- **FLOPs input sizes** when a paper states only "standard": 1280×800 for COCO detection,
  512×2048 for ADE20K segmentation.
- **COCO** without a split named is val2017 for evaluation and train2017 for training.
- **PASCAL VOC 2007** without splits named is trained on VOC 2007 trainval
  (`PASCAL VOC 2007 (train + val)`) and evaluated on VOC 2007 test.

A convention never overrides what the paper states.

## What to include

- **Qualifying papers** introduce a new general-purpose vision backbone architecture or
  pretraining method and report ImageNet-1k classification results for their own
  models. Detection and segmentation results are extracted when present but are not
  required (ResNet (RSB) and CoAtNet report classification only).
- **The paper's own models.** Results the paper reports for other models (baselines)
  are extracted only with database access, and only to fill gaps: the model must
  already be in the database and have no stored result with the same settings. Record
  them under the existing pretrained backbone, with this paper as the source. Never
  create a family, backbone or pretrained backbone for a baseline. A different value
  for a result that is already stored is a correction proposal.
- **Check every baseline row.** Look up each baseline in every comparison table
  (classification, robustness sets such as ImageNet-A/R/Sketch/V2, detection and
  segmentation). A result on a dataset, head or setting not yet stored for that model
  is a gap to fill. When you propose a new head or dataset, add the baseline rows that
  use it for models in the database. Baselines not in the database are skipped: the
  original ResNet and DeiT are not `ResNet (RSB)` or `DeiT III`, which are different
  trained models.
- **Best recipe only.** When the contribution is a training procedure reported at
  several budgets (ResNet strikes back's A1/A2/A3), keep only the headline recipe (A1).
- **Main results only.** Include the configurations in the headline comparison tables
  for classification, detection/instance segmentation and semantic segmentation, and
  the paper's own robustness evaluations. Skip ablation-study variants (e.g. SLaK-T
  trained for 120 epochs), resolution sweeps in analysis figures and sections, and
  results on corrupted versions of downstream datasets.
- **Evaluation resolutions.** A result at a resolution the model was not trained at is
  included when it appears in a main table, with the fine-tune fields left empty.
- **Datasets.** Classification on ImageNet-1k and its robustness/variant sets
  (ImageNet-A, -R, -Sketch, -C, -C-bar, -V2, -ReaL). Detection and instance
  segmentation on COCO val (test-dev only when it is the paper's main number) and
  PASCAL VOC 2007. Semantic segmentation on ADE20K val, Cityscapes and PASCAL VOC when
  reported. Skip classification transfer datasets (CIFAR, iNaturalist, Places,
  Flowers, ...) and panoptic segmentation.
- **Unusual training data.** Models trained on private or composite datasets (e.g.
  InternImage-H on joint semi-labelled data, RepLKNet-XL on MegData73M) are included,
  with the dataset as a new-dataset proposal when it is not in the database.
- **New heads and datasets** (e.g. Sparse R-CNN, ATSS, Objects365): extract them with a
  proposed new head or dataset; they need the maintainer's approval, like categories.

## Families and backbones

- **Family name**: the short name the paper uses for its models, without generic words
  like "Transformer" or "Network" (`Swin`, not "Swin Transformer"; `ConvNeXt V2`,
  `DAT++`).
  One paper can define several families (MetaFormer: IdentityFormer, RandFormer,
  ConvFormer, CAFormer).
- **A family is one architecture.** When a paper applies its idea to several base
  architectures, each base is its own family, named with the paper's labels. LocalViT
  adds its block to DeiT, T2T-ViT, TNT, PVT and Swin: `LocalViT` (the DeiT-based T and
  S), `LocalViT-T2T`, `LocalViT-TNT`, `LocalViT-PVT` and `LocalViT-Swin` (Swin-M and
  Swin). `model_type`, `hierarchical`, `spiking` and `pretrain_method` must be true of
  every model in the family; never choose the majority value for a mixed group.
- **Backbone name**: `Family-Size` using the paper's size labels (`ConvNeXt V2-T`,
  `ResNet-50 (RSB)`, `CoAtNet-3`). Include every size in the main results.
- **Parameters**: millions, backbone only (never the detector or segmenter).
- **model_type** (token mixer):
  - `Attention`, `Convolution`, `State Space Model`, `Identity`, `Random`, or a hybrid.
  - `Attn + Conv`: attention plus a convolution that mixes spatial tokens inside the
    block, whether in separate stages (CoAtNet, MaxViT, CAFormer), interleaved (Iwin),
    as a side branch (RMT) or inside the FFN (TransNeXt). Convolutional position
    encodings (CSWin, BiFormer), stems and downsampling layers do not count.
  - `Conv + SSM`: the same rule with a state-space model in place of attention.
  - `Attn + SSM`: attention and a state-space model both mixing tokens (A2Mamba).
  - A combination with no existing category is a new category proposal.
- **hierarchical**: true for multi-stage models that reduce spatial resolution between
  stages; false for isotropic models (DeiT III, Vim, PlainMamba).
- **spiking**: `true` for spiking neural networks, whose neurons (LIF, IF and similar)
  pass binary or discrete spikes over timesteps (Spikformer, Spike-driven Transformer),
  including ANN-to-SNN conversions reported as SNNs. Quantized or binarized ANNs are not
  spiking. Include the field only when it is true; omit it otherwise.
- **pretrain_method** (family): the method of the family's main models (`Supervised`
  for almost all).

## Pretrained backbones

- One per distinct pretraining configuration of a backbone (dataset, method, epochs,
  resolution).
- **Name**: `Backbone-Data` with `IN1k`, `IN22k`, `JFT300M`, `JFT3B`, and a suffix for a
  variant: `-TL` token labelling, `-LS` long-sequence fine-tuning, `-PTRA`, `-E150`
  (epochs when two configurations differ only in length).
  E.g. `ConvNeXt-B-IN22k`, `FAN-S-Hybrid-IN1k-TL`, `Vim-S-IN1k-LS`, `CoAtNet-4-IN22k-PTRA-E150`.
- A variant made by an extra training stage after pretraining that the paper presents
  as its own model (e.g. Vim's long-sequence models, marked † in its tables) is a
  separate pretrained backbone with a suffix, and its results also record the extra
  stage in their fine-tune fields.
- **pretrain_method**: `Supervised`, `Sup. + TL` (token labelling), `FCMAE`, `MAE`, `CL`,
  `MAP`. Anything else is a new category proposal.
- **pretrain_epochs / resolution**: of the pretraining stage (300 epochs at 224 for most
  ImageNet-1k models, 90 for ImageNet-22k).

## Classification results

- **top_1**; top_5 when reported. For ImageNet-C, top_1 holds mCE; for ImageNet-C-bar,
  CE.
- **gflops**: backbone only, at the evaluation resolution.
- **Fine-tuning** fields (dataset, epochs, resolution) are filled when the evaluation
  dataset or resolution differs from pretraining (typically 30 epochs on ImageNet-1k
  after ImageNet-22k pretraining, or when moving from 224 to 384), when the pretraining
  method differs from the evaluation training (MAE/FCMAE fine-tuning), and when the
  paper describes a separate training stage after pretraining at the same dataset and
  resolution (Vim's long-sequence fine-tuning, a distillation stage). Not for a single
  continuous training run.
- **Intermediate fine-tuning**: the middle stage of a three-stage recipe (e.g. FCMAE on
  ImageNet-22k → supervised ImageNet-22k → ImageNet-1k).

## Detection and instance segmentation

- One result per head, schedule and task: box AP is Object Detection, mask AP is
  Instance Segmentation.
- **Epochs**: named schedules by name, otherwise the epochs the paper states or a
  conversion from iterations (see Conventions).
- **gflops**: backbone + head, at the paper's stated input size (usually 1280×800).
- AP50/AP75 and small/medium/large AP when reported.
- **Intermediate training**: e.g. Objects365 before COCO (a new dataset proposal).

## Semantic segmentation

- **Epochs** from iterations × global batch size ÷ dataset size (see Conventions for
  dataset sizes and the standard batch size).
- **crop_size**: the training crop; see Conventions when it is unstated.
- ss_ and ms_ metrics for single- and multi-scale evaluation; `flip_test` true only when
  flipping is stated for that evaluation.
- **gflops**: backbone + head, at the paper's stated input size (usually 512×2048).
- Semantic FPN is recorded as Panoptic FPN.

## Throughput

- Include when the GPU is stated; leave precision empty if it is not. Include batch
  size when stated.
- GPU and precision must be one of the listed values (`V100`, not "V100 32GB"; `AMP`
  for mixed precision).
- Record it on the backbone for classification throughput, and on the result for
  detection or segmentation throughput.

## Ambiguous figures

- When the paper gives two versions of a number, use the one comparable with other
  models, and note the other.
- When the paper gives the same number at different precisions (e.g. 29M in the main
  table, 28.6M in an appendix), use the more precise one.
- Sparse weights (e.g. SLaK's dynamically sparse kernels): store the dense parameters
  and FLOPs, which count every weight including zeros. That is what ordinary GPUs
  compute; the sparsity-aware figures apply only to hardware that skips unstructured
  zeros. Put the sparsity-aware figures in the note. Sparse attention (attending to a
  subset of tokens, e.g. BiFormer) has a single figure and is unaffected.
- A result at a resolution the model was not pretrained at, when the paper does not say
  how it was obtained (e.g. SLaK-B at 384), is uncertain: the fine-tune fields cannot
  be established.

<!-- prompt:end -->

## Existing records

- When a paper disagrees with a stored value, propose a correction (before, after,
  citation) for approval; never change stored data automatically.
