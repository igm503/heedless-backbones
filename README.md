# [Heedless Backbones](https://heedlessbackbones.com)

![Alt text](assets/plot_view.png?raw=true "Plot View")
A simple web application for comparing computer vision backbones performance on classification and downstream tasks.

## The Problem

Paperswithcode's basic data models and user interface aren't useful either for researchers or industry users interested in comparing the performance of different computer vision backbones for different tasks. The (visible) data model doesn't include:

- Model Family, Model, What Dead was used for the downstream task (e.g. object detection) or What Backbone was used
- What pretraining dataset was used (e.g. IN-1K, IN-21k)
- Details of the pretraining, finetuning, or downstream training
- Throughput, and sometimes even GFLOPS and the number of parameters

This means, for example, that you can't easily:

- Compare the performance of different model families (e.g. compare the Swin and ConvNeXt families)
- Compare model accuracy on multiple tasks
- Do apples-to-apples accuracy comparison, even on one dataset and one task

In addition, the user interface doesn't allow for interesting queries (e.g. what's the best model on ImageNet that can do better than 1000 fps on a V100 with AMP?), and the database is inconsistently maintained.

Heedless Backbones is an attempt to address these shortcomings of Paperswithcode within the space of computer vision backbones. It is built on a data model that treats pretrained foundation models as first class citizens and because of this allows you to make fairly complicated, interesting visualizations of model performance on different tasks. In addition, for now, I will be solely responsible for entering the data, meaning that while it may take a while before the model you're interested in shows up, once it does, it will have far more metadata than any corresponding entry in Paperswithcode.

## Paper ingestion

New models are added by a [paper ingestion job](docs/ingestion.md): Arxiv Troller
discovery, then local Claude Code sessions that screen abstracts and read shortlisted
papers in full, following the [data entry guide](docs/data-entry-guide.md). The agent
checks the paper's official repository, fills gaps in other models' results, validates
its own output against the paper text, and submits evidence-backed records through an
audited importer. Uncertain or conflicting data waits for review on a page in the admin
that shows each value next to a crop of its source. Every publish opens a pull request for
the family with its reference YAML in `family_data/`, a regenerated `db.json`, and a row in
the [model table](#models). The job runs manually, or on a Mac with `deploy/local-agent`.

Hand-written family files can still be imported with `python manage.py add_yaml <file>`.

## Deployment

I'm running this on a cheap digital ocean server, and you can access it by clicking the headline link or by navigating to [heedlessbackbones.com](https://heedlessbackbones.com) in your browser. If for some reason you want to deploy this yourself, you can follow the instructions [here](https://github.com/igm503/django-deploy/blob/main/README.md)

## Tests

With the project dependencies installed, run:

```sh
python django/manage.py test stats ingestion --settings=heedless-backbones.test_settings
```

The tests load `db.json` into an isolated in-memory SQLite database. They cover
plot and table rendering, task comparisons, filters, throughput, source links,
and database query budgets. They do not connect to the database configured in `.env`.

## TODO

- Filter by prominence (since there will soon be too many models)
- Comparison of Heads
- Comparison of Pretraining Datasets
- Better Handling of Queries with no Results

## Completed

- Dynamic Plotting
  - Result vs num params or GFLOPS
  - Result vs Result (Different Dataset or Metric)
  - Result vs Throughput
  - Result vs Pub Date
  - Head, Resolution, Pretrain Dataset, and Pretrain Method Filters
  - Legend Customization
- Accompanying Tables
- Model Family Pages
- Downstream Head Pages
- Dataset Pages
- List Pages (Models, Heads, Datasets)
- LLM-Assisted Data Gen
- Models: see [Models](#models)

## Models

| Model | Paper | Added |
|---|---|---|
| ConvNeXt | [arXiv 2201.03545](https://arxiv.org/abs/2201.03545) | 2024-09-03 |
| TransNeXt | [arXiv 2311.17132](https://arxiv.org/abs/2311.17132) | 2024-09-03 |
| Swin | [arXiv 2103.14030](https://arxiv.org/abs/2103.14030) | 2024-09-23 |
| DeiT III | [arXiv 2204.07118](https://arxiv.org/abs/2204.07118) | 2024-09-23 |
| ConvNeXt V2 | [arXiv 2301.00808](https://arxiv.org/abs/2301.00808) | 2024-09-23 |
| ResNet (RSB) | [arXiv 2110.00476](https://arxiv.org/abs/2110.00476) | 2024-09-23 |
| Hiera | [arXiv 2306.00989](https://arxiv.org/abs/2306.00989) | 2024-09-25 |
| FocalNet | [arXiv 2203.11926](https://arxiv.org/abs/2203.11926) | 2024-09-29 |
| InternImage | [arXiv 2211.05778](https://arxiv.org/abs/2211.05778) | 2024-09-29 |
| CSWin | [arXiv 2107.00652](https://arxiv.org/abs/2107.00652) | 2024-09-29 |
| IdentityFormer | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| RandFormer | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| ConvFormer | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| CAFormer | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| MaxViT | [arXiv 2204.01697](https://arxiv.org/abs/2204.01697) | 2024-09-29 |
| MogaNet | [arXiv 2211.03295](https://arxiv.org/pdf/2211.03295) | 2024-09-29 |
| CoAtNet | [arXiv 2106.04803](https://arxiv.org/abs/2106.04803) | 2024-09-29 |
| VMamba | [arXiv 2401.10166](https://arxiv.org/abs/2401.10166) | 2024-10-14 |
| UniRepLKNet | [arXiv 2410.08049](https://arxiv.org/abs/2410.08049) | 2024-10-16 |
| FAN | [arXiv 2204.12451](https://arxiv.org/abs/2204.12451) | 2025-04-14 |
| SLaK | [arXiv 2207.03620](https://arxiv.org/abs/2207.03620) | 2025-04-15 |
| RepLKNet | [arXiv 2203.06717](https://arxiv.org/abs/2203.06717) | 2025-04-15 |
| BiFormer | [arXiv 2303.08810](https://arxiv.org/abs/2303.08810) | 2025-04-20 |
| MambaOut | [arXiv 2405.07992](https://arxiv.org/abs/2405.07992) | 2025-04-20 |
| GroupMamba | [arXiv 2407.13772](https://arxiv.org/abs/2407.13772) | 2025-04-20 |
| Vim | [arXiv 2401.09417](https://arxiv.org/abs/2401.09417) | 2025-04-20 |
| Hier-Vim | [arXiv 2401.09417](https://arxiv.org/abs/2401.09417) | 2025-04-20 |
| PlainMamba | [arXiv 2403.17695](https://arxiv.org/abs/2403.17695) | 2025-04-20 |
| LocalVim | [arXiv 2403.09338](https://arxiv.org/abs/2403.09338) | 2025-04-20 |
| LocalVMamba | [arXiv 2403.09338](https://arxiv.org/abs/2403.09338) | 2025-04-20 |
| EfficientVMamba | [arXiv 2403.09977](https://arxiv.org/pdf/2403.09977) | 2025-04-20 |
| DAMamba | [arXiv 2502.12627](https://arxiv.org/abs/2502.12627) | 2025-04-20 |
| VSSD | [arXiv 2407.18559](https://arxiv.org/abs/2407.18559) | 2025-04-20 |
| RMT | [arXiv 2309.11523](https://arxiv.org/abs/2309.11523) | 2025-04-20 |
| DAT++ | [arXiv 2309.01430](https://arxiv.org/abs/2309.01430) | 2025-04-20 |
| FAN STL | [arXiv 2401.03844](https://arxiv.org/pdf/2401.03844) | 2025-04-22 |
| DAT | [arXiv 2201.00520](https://arxiv.org/abs/2201.00520) | 2025-04-22 |
| NAT | [arXiv 2204.07143](https://arxiv.org/abs/2204.07143) | 2025-04-22 |
| CoCA ViT | [arXiv 2508.05307](https://arxiv.org/abs/2508.05307) | 2025-11-08 |
| FractalMamba++ | [arXiv 2505.14062](https://arxiv.org/abs/2505.14062) | 2025-11-08 |
| HybridNet | [arXiv 2410.00871](https://arxiv.org/abs/2410.00871) | 2025-11-08 |
| InceptionMamba | [arXiv 2506.08735](https://arxiv.org/abs/2506.08735) | 2025-11-08 |
| Iwin Transformer | [arXiv 2507.18405](https://arxiv.org/abs/2507.18405) | 2025-11-08 |
| MA ViT | [arXiv 2507.00698](https://arxiv.org/html/2507.00698v3) | 2025-11-08 |
| AnchorFormer | [arXiv 2505.16463](https://arxiv.org/abs/2505.16463) | 2025-11-08 |
| Mamba-Adaptor | [arXiv 2505.12685](https://arxiv.org/abs/2505.12685) | 2025-11-08 |
| RecNeXt | [arXiv 2412.19628](https://arxiv.org/abs/2412.19628) | 2025-11-08 |
| S2AFormer | [arXiv 2505.22195](https://arxiv.org/abs/2505.22195) | 2025-11-08 |
| SSViT | [arXiv 2405.13335](https://arxiv.org/abs/2405.13335v1) | 2025-11-08 |
| A2Mamba | [arXiv 2507.16624](https://arxiv.org/abs/2507.16624) | 2025-11-08 |
| SpaRTAN | [arXiv 2507.10999](https://arxiv.org/abs/2507.10999) | 2025-11-08 |
| UniConvNet | [arXiv 2508.09000](https://arxiv.org/abs/2508.09000) | 2025-11-08 |
| UniNeXt | [arXiv 2304.13700](https://arxiv.org/abs/2304.13700) | 2025-11-08 |
| VCMamba | [arXiv 2509.04669](https://arxiv.org/abs/2509.04669) | 2025-11-08 |
| VMINet | [arXiv 2501.02040](https://arxiv.org/abs/2501.02040) | 2025-11-08 |
| GG | [arXiv 2106.02277](https://arxiv.org/abs/2106.02277) | 2026-09-26 |
| LocalViT | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-PVT | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-Swin | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-T2T | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-TNT | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| Shuffle | [arXiv 2106.03650](https://arxiv.org/abs/2106.03650) | 2026-09-26 |

## Updates

- 9-26-2026: added Shuffle
- 9-26-2026: added LocalViT-TNT
- 9-26-2026: added LocalViT-T2T
- 9-26-2026: added LocalViT-Swin
- 9-26-2026: added LocalViT-PVT
- 9-26-2026: added LocalViT
- 9-26-2026: added GG
- 11-8-2025: added CoCAViT, FractalMamba++, HybridNet, InceptionMamba, Iwin, MAViT, AnchorFormer, Mamba-Adaptor, RecNeXt, S2AFormer, SSViT, A2Mamba, SpaRTAN, UniConvNet, UniNeXt, VCMamba, VMINet
- 4-22-2025: added FAN STL, DAT, NAT
- 4-20-2025: added BiFormer, MambaOut, GroupMamba, Vim/Hier-Vim, PlainMamba, LocalVim/LocalVMamba, EfficientVMamba, DAMamba, VSSD, RMT, DAT++
- 4-15-2025: added SLaK, RepLKNet
- 4-14-2025: added FAN
- 10-27-2024: improved plot defaults; bug fixes
- 10-16-2024: added ADE20k results for remaining models; added UniRepLKNet
- 10-15-2024: added ADE20k results for some models; added Semantic Seg to site interface
- 10-13-2024: added Semantic Segmentation task; added VMamba
- 10-9-2024: Website is live
- 9-29-2024: added LLM db data gen command; added benchmark for LLM db data gen; added InternImage, FocalNet, CSwin, RandFormer, IdentityFormer, ConvFormer, CAFormer, MaxViT, MogaNet, CoAtNet
- 9-24-2024: added Hiera
- 9-22-2024: added several models; more train info in tables; refactor plot and table gen; bug fixes
- 9-9-2024: added list pages (models, heads, datasets); added postgresql sql dump to repo
- 9-8-2024: added legend customization for plots; pub date axis option; pretrain method filter; table order defaults
- 9-7-2024: added dataset page; added title bar; improved links and table gen
- 9-6-2024: added downstream head page
- 9-4-2024: added model family page
- 9-2-2024: added resolution and pretraining dataset filters
- 9-1-2024: added table on plot page
