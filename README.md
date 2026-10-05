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
that shows each value next to a crop of its source. Publishing imports the approved data
into the site's database; the pull request records that published data in the repository.
The job runs manually, or on a Mac with `deploy/local-agent`.

Publications accumulate in one automated pull request with each family's reference YAML
in `family_data/`, one regenerated `db.json`, and updates to the [model table](#models)
and About page. Later publications append ordinary commits to the same PR. Its title
summarizes the batch (for example, “Add 5 backbone families; update 3”), and its description
lists every family and paper. Merge the batch when ready; the next publication starts a
fresh branch and PR. Changes to `main` are merged into the pending batch without
force-pushing it.

Use **Refresh records PR** on the ingestion review page to refresh the pending batch.
Model rows and dated updates stay newest-first, and families are alphabetized within
each date in the README. See [Publishing and records](docs/ingestion.md#publishing-and-records)
for the title rules and refresh command.

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

**Models** counts distinct backbone variants (such as Tiny, Small, and Base), not
pretrained checkpoints. Paper-specific entries count variants with checkpoints from that paper.

| Model | Models | Paper | Added |
|---|---:|---|---|
| E-SpikeFormer+Adaptive PT-SSA | 1 | [arXiv 2610.03291](https://arxiv.org/abs/2610.03291) | 2026-10-05 |
| ConvNeXt-dcls | 3 | [arXiv 2112.03740](https://arxiv.org/abs/2112.03740) | 2026-10-03 |
| CSKAFormer | 2 | [arXiv 2412.07049](https://arxiv.org/abs/2412.07049) | 2026-10-03 |
| DeBiFormer | 3 | [arXiv 2410.08582](https://arxiv.org/abs/2410.08582) | 2026-10-03 |
| EViT | 4 | [arXiv 2310.06629](https://arxiv.org/abs/2310.06629) | 2026-10-03 |
| FaViT | 4 | [arXiv 2312.08614](https://arxiv.org/abs/2312.08614) | 2026-10-03 |
| FST | 1 | [arXiv 2609.38348](https://arxiv.org/abs/2609.38348) | 2026-10-03 |
| FViT | 4 | [arXiv 2402.11303](https://arxiv.org/abs/2402.11303) | 2026-10-03 |
| Pale | 3 | [arXiv 2112.14000](https://arxiv.org/abs/2112.14000) | 2026-10-03 |
| QKFormer+SCA | 2 | [arXiv 2610.01403](https://arxiv.org/abs/2610.01403) | 2026-10-03 |
| ResT | 4 | [arXiv 2105.13677](https://arxiv.org/abs/2105.13677) | 2026-10-03 |
| SDT-V1+SCA | 2 | [arXiv 2610.01403](https://arxiv.org/abs/2610.01403) | 2026-10-03 |
| SDT-V3+SCA | 3 | [arXiv 2610.01403](https://arxiv.org/abs/2610.01403) | 2026-10-03 |
| SKAFormer | 2 | [arXiv 2412.07049](https://arxiv.org/abs/2412.07049) | 2026-10-03 |
| TransXNet | 3 | [arXiv 2310.19380](https://arxiv.org/abs/2310.19380) | 2026-10-03 |
| VBB | 2 | [arXiv 2311.05988](https://arxiv.org/abs/2311.05988) | 2026-10-03 |
| Win | 3 | [arXiv 2211.14255](https://arxiv.org/abs/2211.14255) | 2026-10-03 |
| SD-Transformer+k-WTA Router | 2 | [arXiv 2610.01418](https://arxiv.org/abs/2610.01418) | 2026-10-02 |
| Spikformer+k-WTA Router | 2 | [arXiv 2610.01418](https://arxiv.org/abs/2610.01418) | 2026-10-02 |
| Spikingformer+k-WTA Router | 2 | [arXiv 2610.01418](https://arxiv.org/abs/2610.01418) | 2026-10-02 |
| GSAP | 3 | [arXiv 2609.26297](https://arxiv.org/abs/2609.26297) | 2026-09-29 |
| iFormer (Mobile) | 6 | [arXiv 2501.15369](https://arxiv.org/abs/2501.15369) | 2026-09-29 |
| L2ViT | 3 | [arXiv 2501.16182](https://arxiv.org/abs/2501.16182) | 2026-09-29 |
| LSNet | 3 | [arXiv 2503.23135](https://arxiv.org/abs/2503.23135) | 2026-09-29 |
| MVFormer | 3 | [arXiv 2411.18995](https://arxiv.org/abs/2411.18995) | 2026-09-29 |
| PPMA | 3 | [arXiv 2506.15940](https://arxiv.org/abs/2506.15940) | 2026-09-29 |
| TinyViM | 3 | [arXiv 2411.17473](https://arxiv.org/abs/2411.17473) | 2026-09-29 |
| V2M + local window | 2 | [arXiv 2410.10382](https://arxiv.org/abs/2410.10382) | 2026-09-29 |
| V2M* | 2 | [arXiv 2410.10382](https://arxiv.org/abs/2410.10382) | 2026-09-29 |
| FAT | 5 | [arXiv 2306.00396](https://arxiv.org/abs/2306.00396) | 2026-09-28 |
| FMViT | 5 | [arXiv 2311.05707](https://arxiv.org/abs/2311.05707) | 2026-09-28 |
| LaViT | 3 | [arXiv 2406.00427](https://arxiv.org/abs/2406.00427) | 2026-09-28 |
| Mamba-R | 4 | [arXiv 2405.14858](https://arxiv.org/abs/2405.14858) | 2026-09-28 |
| MSVMamba | 3 | [arXiv 2405.14174](https://arxiv.org/abs/2405.14174) | 2026-09-28 |
| RepNeXt | 6 | [arXiv 2406.16004](https://arxiv.org/abs/2406.16004) | 2026-09-28 |
| RepViT | 5 | [arXiv 2307.09283](https://arxiv.org/abs/2307.09283) | 2026-09-28 |
| SW | 2 | [arXiv 2401.12736](https://arxiv.org/abs/2401.12736) | 2026-09-28 |
| WTConvNeXt | 3 | [arXiv 2407.05848](https://arxiv.org/abs/2407.05848) | 2026-09-28 |
| CE-CSWin | 3 | [arXiv 2207.13317](https://arxiv.org/abs/2207.13317) | 2026-09-27 |
| CE-CvT | 1 | [arXiv 2207.13317](https://arxiv.org/abs/2207.13317) | 2026-09-27 |
| CE-PVT | 1 | [arXiv 2207.13317](https://arxiv.org/abs/2207.13317) | 2026-09-27 |
| CE-Swin | 3 | [arXiv 2207.13317](https://arxiv.org/abs/2207.13317) | 2026-09-27 |
| CETNet | 3 | [arXiv 2207.13317](https://arxiv.org/abs/2207.13317) | 2026-09-27 |
| CloFormer | 3 | [arXiv 2303.17803](https://arxiv.org/abs/2303.17803) | 2026-09-27 |
| iFormer | 3 | [arXiv 2205.12956](https://arxiv.org/abs/2205.12956) | 2026-09-27 |
| InceptionNeXt | 4 | [arXiv 2303.16900](https://arxiv.org/abs/2303.16900) | 2026-09-27 |
| InceptionNeXt (iso.) | 1 | [arXiv 2303.16900](https://arxiv.org/abs/2303.16900) | 2026-09-27 |
| LightViT | 3 | [arXiv 2207.05557](https://arxiv.org/abs/2207.05557) | 2026-09-27 |
| LinGlo | 3 | [arXiv 2207.00188](https://arxiv.org/abs/2207.00188) | 2026-09-27 |
| SepViT | 4 | [arXiv 2203.15380](https://arxiv.org/abs/2203.15380) | 2026-09-27 |
| STViT | 3 | [arXiv 2211.11167](https://arxiv.org/abs/2211.11167) | 2026-09-27 |
| SwiftFormer | 4 | [arXiv 2303.15446](https://arxiv.org/abs/2303.15446) | 2026-09-27 |
| CMT | 5 | [arXiv 2107.06263](https://arxiv.org/abs/2107.06263) | 2026-09-26 |
| GG | 2 | [arXiv 2106.02277](https://arxiv.org/abs/2106.02277) | 2026-09-26 |
| HAT-Net | 4 | [arXiv 2106.03180](https://arxiv.org/abs/2106.03180) | 2026-09-26 |
| LocalViT | 2 | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-PVT | 1 | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-Swin | 2 | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-T2T | 1 | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| LocalViT-TNT | 1 | [arXiv 2104.05707](https://arxiv.org/abs/2104.05707) | 2026-09-26 |
| P2T | 4 | [arXiv 2106.12011](https://arxiv.org/abs/2106.12011) | 2026-09-26 |
| Shuffle | 3 | [arXiv 2106.03650](https://arxiv.org/abs/2106.03650) | 2026-09-26 |
| A2Mamba | 5 | [arXiv 2507.16624](https://arxiv.org/abs/2507.16624) | 2025-11-08 |
| AnchorFormer | 3 | [arXiv 2505.16463](https://arxiv.org/abs/2505.16463) | 2025-11-08 |
| CoCA ViT | 3 | [arXiv 2508.05307](https://arxiv.org/abs/2508.05307) | 2025-11-08 |
| FractalMamba++ | 3 | [arXiv 2505.14062](https://arxiv.org/abs/2505.14062) | 2025-11-08 |
| HybridNet | 4 | [arXiv 2410.00871](https://arxiv.org/abs/2410.00871) | 2025-11-08 |
| InceptionMamba | 3 | [arXiv 2506.08735](https://arxiv.org/abs/2506.08735) | 2025-11-08 |
| Iwin Transformer | 4 | [arXiv 2507.18405](https://arxiv.org/abs/2507.18405) | 2025-11-08 |
| MA ViT | 4 | [arXiv 2507.00698](https://arxiv.org/html/2507.00698v3) | 2025-11-08 |
| Mamba-Adaptor | 2 | [arXiv 2505.12685](https://arxiv.org/abs/2505.12685) | 2025-11-08 |
| RecNeXt | 15 | [arXiv 2412.19628](https://arxiv.org/abs/2412.19628) | 2025-11-08 |
| S2AFormer | 5 | [arXiv 2505.22195](https://arxiv.org/abs/2505.22195) | 2025-11-08 |
| SpaRTAN | 2 | [arXiv 2507.10999](https://arxiv.org/abs/2507.10999) | 2025-11-08 |
| SSViT | 4 | [arXiv 2405.13335](https://arxiv.org/abs/2405.13335v1) | 2025-11-08 |
| UniConvNet | 13 | [arXiv 2508.09000](https://arxiv.org/abs/2508.09000) | 2025-11-08 |
| UniNeXt | 3 | [arXiv 2304.13700](https://arxiv.org/abs/2304.13700) | 2025-11-08 |
| VCMamba | 3 | [arXiv 2509.04669](https://arxiv.org/abs/2509.04669) | 2025-11-08 |
| VMINet | 4 | [arXiv 2501.02040](https://arxiv.org/abs/2501.02040) | 2025-11-08 |
| DAT | 3 | [arXiv 2201.00520](https://arxiv.org/abs/2201.00520) | 2025-04-22 |
| FAN STL | 4 | [arXiv 2401.03844](https://arxiv.org/pdf/2401.03844) | 2025-04-22 |
| NAT | 4 | [arXiv 2204.07143](https://arxiv.org/abs/2204.07143) | 2025-04-22 |
| BiFormer | 3 | [arXiv 2303.08810](https://arxiv.org/abs/2303.08810) | 2025-04-20 |
| DAMamba | 3 | [arXiv 2502.12627](https://arxiv.org/abs/2502.12627) | 2025-04-20 |
| DAT++ | 3 | [arXiv 2309.01430](https://arxiv.org/abs/2309.01430) | 2025-04-20 |
| EfficientVMamba | 3 | [arXiv 2403.09977](https://arxiv.org/pdf/2403.09977) | 2025-04-20 |
| GroupMamba | 3 | [arXiv 2407.13772](https://arxiv.org/abs/2407.13772) | 2025-04-20 |
| Hier-Vim | 3 | [arXiv 2401.09417](https://arxiv.org/abs/2401.09417) | 2025-04-20 |
| LocalVim | 2 | [arXiv 2403.09338](https://arxiv.org/abs/2403.09338) | 2025-04-20 |
| LocalVMamba | 2 | [arXiv 2403.09338](https://arxiv.org/abs/2403.09338) | 2025-04-20 |
| MambaOut | 4 | [arXiv 2405.07992](https://arxiv.org/abs/2405.07992) | 2025-04-20 |
| PlainMamba | 3 | [arXiv 2403.17695](https://arxiv.org/abs/2403.17695) | 2025-04-20 |
| RMT | 4 | [arXiv 2309.11523](https://arxiv.org/abs/2309.11523) | 2025-04-20 |
| Vim | 3 | [arXiv 2401.09417](https://arxiv.org/abs/2401.09417) | 2025-04-20 |
| VSSD | 4 | [arXiv 2407.18559](https://arxiv.org/abs/2407.18559) | 2025-04-20 |
| RepLKNet | 3 | [arXiv 2203.06717](https://arxiv.org/abs/2203.06717) | 2025-04-15 |
| SLaK | 3 | [arXiv 2207.03620](https://arxiv.org/abs/2207.03620) | 2025-04-15 |
| FAN | 8 | [arXiv 2204.12451](https://arxiv.org/abs/2204.12451) | 2025-04-14 |
| UniRepLKNet | 9 | [arXiv 2410.08049](https://arxiv.org/abs/2410.08049) | 2024-10-16 |
| VMamba | 3 | [arXiv 2401.10166](https://arxiv.org/abs/2401.10166) | 2024-10-14 |
| CAFormer | 4 | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| CoAtNet | 8 | [arXiv 2106.04803](https://arxiv.org/abs/2106.04803) | 2024-09-29 |
| ConvFormer | 4 | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| CSWin | 4 | [arXiv 2107.00652](https://arxiv.org/abs/2107.00652) | 2024-09-29 |
| FocalNet | 7 | [arXiv 2203.11926](https://arxiv.org/abs/2203.11926) | 2024-09-29 |
| IdentityFormer | 5 | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| InternImage | 6 | [arXiv 2211.05778](https://arxiv.org/abs/2211.05778) | 2024-09-29 |
| MaxViT | 5 | [arXiv 2204.01697](https://arxiv.org/abs/2204.01697) | 2024-09-29 |
| MogaNet | 6 | [arXiv 2211.03295](https://arxiv.org/pdf/2211.03295) | 2024-09-29 |
| RandFormer | 5 | [arXiv 2210.13452](https://arxiv.org/abs/2210.13452) | 2024-09-29 |
| Hiera | 6 | [arXiv 2306.00989](https://arxiv.org/abs/2306.00989) | 2024-09-25 |
| ConvNeXt V2 | 8 | [arXiv 2301.00808](https://arxiv.org/abs/2301.00808) | 2024-09-23 |
| DeiT III | 4 | [arXiv 2204.07118](https://arxiv.org/abs/2204.07118) | 2024-09-23 |
| ResNet (RSB) | 5 | [arXiv 2110.00476](https://arxiv.org/abs/2110.00476) | 2024-09-23 |
| Swin | 4 | [arXiv 2103.14030](https://arxiv.org/abs/2103.14030) | 2024-09-23 |
| ConvNeXt | 5 | [arXiv 2201.03545](https://arxiv.org/abs/2201.03545) | 2024-09-03 |
| TransNeXt | 4 | [arXiv 2311.17132](https://arxiv.org/abs/2311.17132) | 2024-09-03 |

## Updates

- 10-5-2026: added E-SpikeFormer+Adaptive PT-SSA
- 10-3-2026: added ConvNeXt-dcls, CSKAFormer, DeBiFormer, EViT, FaViT, FST, FViT, Pale, QKFormer+SCA, ResT, SDT-V1+SCA, SDT-V3+SCA, SKAFormer, TransXNet, VBB, Win
- 10-2-2026: added SD-Transformer+k-WTA Router, Spikformer+k-WTA Router, Spikingformer+k-WTA Router
- 9-29-2026: added GSAP, iFormer (Mobile), L2ViT, LSNet, MVFormer, PPMA, TinyViM, V2M + local window, V2M*
- 9-28-2026: added FAT, FMViT, LaViT, Mamba-R, MSVMamba, RepNeXt, RepViT, SW, WTConvNeXt
- 9-27-2026: added CE-CSWin, CE-CvT, CE-PVT, CE-Swin, CETNet, CloFormer, iFormer, InceptionNeXt, InceptionNeXt (iso.), LightViT, LinGlo, SepViT, STViT, SwiftFormer
- 9-26-2026: added CMT, GG, HAT-Net, LocalViT, LocalViT-PVT, LocalViT-Swin, LocalViT-T2T, LocalViT-TNT, P2T, Shuffle
- 11-8-2025: added A2Mamba, AnchorFormer, CoCAViT, FractalMamba++, HybridNet, InceptionMamba, Iwin, Mamba-Adaptor, MAViT, RecNeXt, S2AFormer, SpaRTAN, SSViT, UniConvNet, UniNeXt, VCMamba, VMINet
- 4-22-2025: added DAT, FAN STL, NAT
- 4-20-2025: added BiFormer, DAMamba, DAT++, EfficientVMamba, GroupMamba, LocalVim/LocalVMamba, MambaOut, PlainMamba, RMT, Vim/Hier-Vim, VSSD
- 4-15-2025: added RepLKNet, SLaK
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
