PROMPT = """
Please analyze the provided paper and generate a YAML 
structure describing the new model family, its backbones, 
and its results, and related fps measurements. Since this
YAML will be used to update a database, you should be
aware of the following information:

Table definitions:

{model_definitions}

Available Datasets, Tasks, Downstream Heads, and GPUs:

{static_data}

However, please ignore Panoptic and Semantic Segmentation.

The YAML you will generate will be used to populate to 
create new entries in the tables listed above. An example
of the desired YAML structure to generate the tables is
given below. Note that foreign key and many to many
relationships are indicated by nesting. Note also that
classification results and instance (this means instance
segmentation or object detection) results can take
more than just one metric. For example, instance results 
can take mAP, AP50, AP75, mAPs, mAPm, mAPl, and gflops.
If some of these metrics are not included for a given
experiment, you are welcome to leave the keys out of the 
YAML. However, please include all metrics that are listed.
Finally, please include relevant gflop measurements along
with each result.

The following is an example of the desired YAML structure:

```yaml
name: "ConvNeXt"
model_type: "Convolution"
hierarchical: true
pretrain_method: "Supervised"
pub_date: "2022-03-02"
paper: "https://arxiv.org/abs/2201.03545"
github: "https://github.com/facebookresearch/ConvNeXt"

backbones:
  - name: "ConvNeXt-T"
    m_parameters: 28.6
    fps_measurements:
      - resolution: 224
        fps: 774.7
        gpu: "V100"
        precision: "FP16"
      - resolution: 224
        fps: 1943.5
        gpu: "A100"
        precision: "TF32"
    pretrained_backbones:
      - name: "ConvNeXt-T-IN1k"
        pretrain_dataset: "ImageNet-1K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 300
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 300
            fine_tune_resolution: 224
            top_1: 82.1
            gflops: 4.5
        instance_results:
          - head: "Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Object Detection"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 46.2
            AP50: 67.9
            AP75: 50.8
            gflops: 262
            fps_measurements:
              - resolution: 1280
                fps: 25.6
                gpu: "A100"
                precision: "FP16"
          - head: "Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Instance Segmentation"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 41.7
            AP50: 65.0
            AP75: 44.9
            gflops: 262
            fps_measurements:
              - resolution: 1280
                fps: 25.6
                gpu: "A100"
                precision: "FP16"
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Object Detection"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 50.4
            AP50: 69.1
            AP75: 54.8
            gflops: 741
            fps_measurements:
              - resolution: 1280
                fps: 13.5
                gpu: "A100"
                precision: "FP16"
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Instance Segmentation"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 43.7
            AP50: 66.5
            AP75: 47.3
            gflops: 741
            fps_measurements:
              - resolution: 1280
                fps: 13.5
                gpu: "A100"
                precision: "FP16"
      - name: "ConvNeXt-T-IN22k"
        pretrain_dataset: "ImageNet-22K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 90
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 224
            top_1: 82.9
            gflops: 4.5
          - dataset: "ImageNet-1K"
            resolution: 384
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 384
            top_1: 84.1
            gflops: 13.1

  - name: "ConvNeXt-S"
    m_parameters: 50
    fps_measurements:
      - resolution: 224
        fps: 447.1
        gpu: "V100"
        precision: "FP16"
      - resolution: 224
        fps: 1275.3
        gpu: "A100"
        precision: "TF32"
    pretrained_backbones:
      - name: "ConvNeXt-S-IN1k"
        pretrain_dataset: "ImageNet-1K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 300
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 300
            fine_tune_resolution: 224
            top_1: 83.1
            gflops: 8.7
        instance_results:
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Object Detection"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 51.9
            AP50: 70.8
            AP75: 56.5
            gflops: 827
            fps_measurements:
              - resolution: 1280
                fps: 12.0
                gpu: "A100"
                precision: "FP16"
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Instance Segmentation"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 45.0
            AP50: 68.4
            AP75: 49.1
            gflops: 827
            fps_measurements:
              - resolution: 1280
                fps: 12.0
                gpu: "A100"
                precision: "FP16"
      - name: "ConvNeXt-S-IN22k"
        pretrain_dataset: "ImageNet-22K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 90
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 224
            top_1: 84.6
            gflops: 8.7
          - dataset: "ImageNet-1K"
            resolution: 384
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 384
            top_1: 85.8
            gflops: 25.5

  - name: "ConvNeXt-B"
    m_parameters: 88.6
    fps_measurements:
      - resolution: 224
        fps: 292.1
        gpu: "V100"
        precision: "FP16"
      - resolution: 224
        fps: 969.0
        gpu: "A100"
        precision: "TF32"
      - resolution: 384
        fps: 95.7
        gpu: "V100"
        precision: "FP16"
      - resolution: 384
        fps: 336.6
        gpu: "A100"
        precision: "TF32"
    pretrained_backbones:
      - name: "ConvNeXt-B-IN1k"
        pretrain_dataset: "ImageNet-1K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 300
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 300
            fine_tune_resolution: 224
            top_1: 83.8
            gflops: 15.4
          - dataset: "ImageNet-1K"
            resolution: 384
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 384
            top_1: 85.1
            gflops: 45.0
        instance_results:
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Object Detection"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 52.7
            AP50: 71.3
            AP75: 57.2
            gflops: 964
            fps_measurements:
              - resolution: 1280
                fps: 11.4
                gpu: "A100"
                precision: "FP16"
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Instance Segmentation"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 45.6
            AP50: 68.9
            AP75: 49.5
            gflops: 964
            fps_measurements:
              - resolution: 1280
                fps: 11.4
                gpu: "A100"
                precision: "FP16"
      - name: "ConvNeXt-B-IN22k"
        pretrain_dataset: "ImageNet-22K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 90
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 224
            top_1: 85.8
            gflops: 15.4
          - dataset: "ImageNet-1K"
            resolution: 384
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 384
            top_1: 86.8
            gflops: 45.1
        instance_results:
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Object Detection"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 54.0
            AP50: 73.1
            AP75: 58.8
            gflops: 964
            fps_measurements:
              - resolution: 1280
                fps: 11.5
                gpu: "A100"
                precision: "FP16"
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Instance Segmentation"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 46.9
            AP50: 70.6
            AP75: 51.3
            gflops: 964
            fps_measurements:
              - resolution: 1280
                fps: 11.5
                gpu: "A100"
                precision: "FP16"

  - name: "ConvNeXt-L"
    m_parameters: 197.8
    fps_measurements:
      - resolution: 224
        fps: 146.8
        gpu: "V100"
        precision: "FP16"
      - resolution: 224
        fps: 611.5
        gpu: "A100"
        precision: "TF32"
      - resolution: 384
        fps: 50.4
        gpu: "V100"
        precision: "FP16"
      - resolution: 384
        fps: 211.4
        gpu: "A100"
        precision: "TF32"
    pretrained_backbones:
      - name: "ConvNeXt-L-IN1k"
        pretrain_dataset: "ImageNet-1K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 300
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 300
            fine_tune_resolution: 224
            top_1: 84.3
            gflops: 34.4
          - dataset: "ImageNet-1K"
            resolution: 384
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 384
            top_1: 85.5
            gflops: 101.0
      - name: "ConvNeXt-L-IN22k"
        pretrain_dataset: "ImageNet-22K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 90
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 224
            top_1: 86.6
            gflops: 34.4
          - dataset: "ImageNet-1K"
            resolution: 384
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 384
            top_1: 87.5
            gflops: 101.0
        instance_results:
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Object Detection"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 54.8
            AP50: 73.8
            AP75: 59.8
            gflops: 1354
            fps_measurements:
              - resolution: 1280
                fps: 10.0
                gpu: "A100"
                precision: "FP16"
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Instance Segmentation"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 47.6
            AP50: 71.3
            AP75: 51.7
            gflops: 1354
            fps_measurements:
              - resolution: 1280
                fps: 10.0
                gpu: "A100"
                precision: "FP16"

  - name: "ConvNeXt-XL"
    m_parameters: 350.2
    fps_measurements:
      - resolution: 224
        fps: 424.4
        gpu: "A100"
        precision: "TF32"
      - resolution: 384
        fps: 147.4
        gpu: "A100"
        precision: "TF32"
    pretrained_backbones:
      - name: "ConvNeXt-XL-IN22k"
        pretrain_dataset: "ImageNet-22K"
        pretrain_method: "Supervised"
        pretrain_resolution: 224
        pretrain_epochs: 90
        classification_results:
          - dataset: "ImageNet-1K"
            resolution: 224
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 224
            top_1: 87.0
            gflops: 60.9
          - dataset: "ImageNet-1K"
            resolution: 384
            fine_tune_dataset: "ImageNet-1K"
            fine_tune_epochs: 30
            fine_tune_resolution: 384
            top_1: 87.8
            gflops: 179.0
        instance_results:
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Object Detection"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 55.2
            AP50: 74.2
            AP75: 59.9
            gflops: 1898
            fps_measurements:
              - resolution: 1280
                fps: 8.6
                gpu: "A100"
                precision: "FP16"
          - head: "Cascade Mask R-CNN"
            dataset: "COCO (val)"
            instance_type: "Instance Segmentation"
            train_dataset: "COCO (train)"
            train_epochs: 36
            mAP: 47.7
            AP50: 71.6
            AP75: 52.2
            gflops: 1898
            fps_measurements:
              - resolution: 1280
                fps: 8.6
                gpu: "A100"
                precision: "FP16"
```


- Include every backbone from the model family introduced in the paper. 
- For each bacbkbone, include every pretrained backbone introduced in the 
paper.
- Include every image classification result for the pretrained backbones 
- Include every object detection and instance segmentation result for 
the pretrained backbones, provided the downstream head used in the task
exists in the list given above. 
- Sometimes not all of the pretrained backbones will have instance results,
or different sizes will use different heads and training schemes. Keep this
in mind, and include all and only those instance results found in the paper.
- Include every FPS result available for the models, including for 
instance tasks (instance segmentation and object detection). 
- FPS measurements are not always available, so the fps fields are optional. Only
include measurements if they are available, and if you can find the gpu + float 
precision used to take the measurement.

The following is the paper content:

{pdf_content}

Please enclose your yaml in tags as follows:
```yaml 
# yaml content 
```
"""
