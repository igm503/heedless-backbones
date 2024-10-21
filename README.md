# [Heedless Backbones](https://heedlessbackbones.com)
![Alt text](assets/plot_view.png?raw=true "Plot View")
A simple web application for comparing computer vision backbones performance on classification and downstream tasks.

## The Problem
Paperswithcode's basic data models and user interface aren't useful either for researchers or industry users interested in comparing the performance of different computer vision backbones for different tasks. The (visible) data model doesn't include:
- Model Family and Model What head was used for the downstream task (e.g. object detection) or what backbone was used
- What pretraining dataset was used (e.g. IN-1K, IN-21k)
- Details of the pretraining, finetuning, or downstream training
- Throughput, and sometimes even GFLOPS and the number of parameters

This means, for example, that you can't easily:
- Compare the performance of different model families (e.g. compare the Swin and ConvNeXt families)
- Compare model accuracy on multiple tasks
- Do apples-to-apples accuracy comparison, even on one dataset and one task

In addition, the user interface doesn't allow for interesting queries (e.g. what's the best model on ImageNet that can do better than 1000 fps on a V100 with AMP?), and the database is inconsistently maintained.

Heedless Backbones is an attempt to address these shortcomings of Paperswithcode within the space of computer vision backbones. It is built on a data model that treats pretrained foundation models as first class citizens and because of this allows you to make fairly complicated, interesting visualizations of model performance on different tasks. In addition, for now, I will be solely responsible for entering the data, meaning that while it may take a while before the model you're interested in shows up, once it does, it will have far more metadata than any corresponding entry in Paperswithcode.

## LLM-Assisted Data Entry
To speed up adding models to the database, I use an LLM (Claude 3.5 Sonnet, currently), to generate yaml files for each model family that I want to add, using research paper pdfs as input. It works alright. If you'd like to use this tool, do the following:

1. copy the ```example.env``` file to ```.env```
2. replace ANTHROPY_API_KEY with your Anthropic API key (you can use the Open AI key variable as well, but you'd need to modify the code in ```django/stats/management/commands/llm_gen.py```)
3. move to the ```django``` directory in your command prompt
4. run ```python manage.py llm_gen [research paper pdf url] [name for the generated yaml file]```

That will produce a yaml file in the ```family_data/``` dir with the name you specified. Once you've looked it over and edited it to your satisfaction, you can add it to the database with 
```
python manage.py add_yaml [name of the generated yaml file]
```
So, for example, if you wanted to add ConvNeXT, you could do the following:
```
cd /path-to-this-repo/django

python manage.py llm_gen https://arxiv.org/pdf/2201.03545 ConvNeXT

# edit family_data/ConvNeXT.yaml to your satisfaction

python manage.py add_yaml ConvNeXT
```

## Deployment
I'm running this on a cheap digital ocean server, and you can access it by clicking the headline link or by navigating to [heedlessbackbones.com](https://heedlessbackbones.com) in your browser. If for some reason you want to deploy this yourself, you can follow the instructions [here](https://github.com/igm503/django-deploy/blob/main/README.md)

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
- Models Added:
  - [ConvNeXt](https://arxiv.org/abs/2201.03545)
  - [TransNeXt](https://arxiv.org/abs/2311.17132)
  - [Swin](https://arxiv.org/abs/2103.14030)
  - [DeiT III](https://arxiv.org/abs/2204.07118)
  - [ConvNextV2](https://arxiv.org/abs/2301.00808)
  - [ResNet (RSB)](https://arxiv.org/abs/2110.00476)
  - [Hiera](https://arxiv.org/abs/2306.00989)
  - [FocalNet](https://arxiv.org/abs/2203.11926)
  - [InternImage](https://arxiv.org/abs/2211.05778)
  - [CSwin](https://arxiv.org/abs/2107.00652)
  - [MetaFormers](https://arxiv.org/abs/2210.13452)
    - IdentityFormer
    - RandFormer
    - ConvFormer
    - CAFormer
  - [MaxViT](https://arxiv.org/abs/2204.01697)
  - [MogaNet](https://arxiv.org/pdf/2211.03295)
  - [CoAtNet](https://arxiv.org/abs/2108.12895)
  - [VMamba](https://arxiv.org/abs/2401.10166)
  - [UniRepLKNet](https://arxiv.org/abs/2311.15599)

## Updates
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
