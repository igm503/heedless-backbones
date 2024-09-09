# [Coming Soon]

# Heedless Backbones
![Alt text](assets/plot_view.png?raw=true "Plot View")
A simple web application for comparing computer vision backbones performance on classification and downstream tasks.

## The Problem
Paperswithcode's basic data models and user interface aren't useful either for researchers or industry users interested in comparing the performance of different computer vision backbones for different tasks. The (visible) data model doesn't include:
- Model Family and Model
- What head was used for the downstream task (e.g. object detection) or what backbone was used
- What pretraining dataset was used (e.g. IN-1K, IN-21k)
- Details of the pretraining, finetuning, or downstream training
- Throughput, and sometimes even GFLOPS and the number of parameters

This means, for example, that you can't easily:
- Compare the performance of different model families (e.g. compare the Swin and ConvNeXt families)
- Compare model accuracy on multiple tasks
- Do apples-to-apples accuracy comparison, even on one dataset and one task

In addition, the user interface doesn't allow for interesting queries (e.g. what's the best model on ImageNet that can do better than 1000 fps on a V100?), and the database is inconsistently maintained.

Heedless Backbones is an attempt to address these shortcomings of Paperswithcode within the space of computer vision backbones. It is built on a data model that treats pretrained foundation models as first class citizens and because of this allows you to make fairly complicated, interesting visualizations of model performance on different tasks. In addition, for now, I will be solely responsible for entering the data, meaning that while it may take a while before the model you're interested in shows up, once it does, it will have far more metadata than any corresponding entry in Paperswithcode.

## TODO
- Add List Pages (datasets, families, heads)
- Host the Website
- Add Semantic Segmentation as a downstream task
- Add More Backbones

## Completed
- Dynamic Plotting
  - Result vs num params or GFLOPS
  - Result vs Result (Different Dataset or Metric)
  - Result vs Throughput
  - Result vs Pub Date
  - Downstream Head, Resolution, Pretrain Dataset, and Pretrain Method Filters 
  - Legend Customization
- Accompanying Tables
- Model Family Pages
- Downstream Head Pages
- Models Added:
  - [ConvNeXt](https://arxiv.org/abs/2201.03545)
  - [TransNeXt](https://arxiv.org/abs/2311.17132)

## Updates
- 9-8-2024: added legend customization for plots; pub date axis option; pretrain method filter; table order defaults
- 9-7-2024: added dataset page; added title bar; improved links and table gen
- 9-6-2024: added downstream head page
- 9-4-2024: added model family page
- 9-2-2024: added resolution and pretraining dataset filters
- 9-1-2024: added table on plot page
