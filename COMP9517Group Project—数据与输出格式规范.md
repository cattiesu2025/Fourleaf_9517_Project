# COMP9517 Group Project — 数据与输出格式规范

本文档是全组的"数据合同"，所有成员必须严格遵守。任何格式变更必须先在群里同步，经A和E确认后再修改，不得私自调整。

---

## 一、项目目录结构

COMP9517_Project/
│
├── data/
│   ├── raw/
│   │   ├── train_mini/
│   │   │   ├── 00000/
│   │   │   ├── 00001/
│   │   │   └── ...
│   │   ├── val/
│   │   ├── train_mini.json
│   │   └── val.json
│   │
│   ├── metadata/
│   │   ├── selected_classes.csv
│   │   ├── class_to_idx.json
│   │   ├── idx_to_class.json
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── longtail_train.csv
│   │   ├── longtail_resampled_train.csv
│   │   ├── split_config.json
│   │   └── longtail_config.json
│   │
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   ├── transforms.py
│   │   └── sampling.py
│   │
│   ├── traditional/
│   │   ├── sift_bovw_svm.py
│   │   ├── hog_random_forest.py
│   │   └── extract_features.py
│   │
│   ├── scratch/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── transfer/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── gradcam.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       ├── metrics.py
│       ├── confusion_matrix.py
│       ├── robustness.py
│       ├── degradation.py        # 实时退化函数，E提供
│       ├── error_analysis.py
│       └── plots.py
│
├── outputs/
│   ├── traditional/
│   ├── scratch/
│   ├── transfer/
│   ├── robustness/
│   └── evaluation/
│       ├── all_methods_comparison.csv     # 所有method的指标汇总表
│       ├── accuracy_vs_time.png           # 报告里的性能-耗时对比图
│       ├── robustness_curves.png          # 各方法在不同退化severity下的曲线对比
│       └── final_report_tables/           # 直接可以贴进报告的表格
│
├── configs/
│   ├── data_config.yaml
│   ├── traditional.yaml
│   ├── scratch.yaml
│   ├── transfer.yaml
│   └── robustness.yaml
│
├── report/
├── presentation/
├── README.md
└── requirements.txt



> ⚠️ **提交注意**：最终代码ZIP不得包含原始图片、模型权重（checkpoint）、大量结果图片，代码包上限 40MB。

---

## 二、数据集划分方案

固定为 **500个随机类别**，每类：

| Split | 每类图片数 | 500类总数 | 来源 |
|---|---:|---:|---|
| Train | 40 | 20,000 | train_mini |
| Validation | 10 | 5,000 | train_mini |
| Test | 10 | 5,000 | **official val** |
| 总计 | 60 | 30,000 | iNat2021 |

**Test 必须来自官方 validation set**，不得从 train_mini 再抽测试集。

---

## 三、Metadata 文件规范

所有人训练/推理时只读取以下文件，**不得自行重新划分数据**。

### 3.1 `train.csv` / `val.csv` / `test.csv`

```csv
image_id,image_path,original_class_id,class_idx,class_name,split
12031,data/raw/train_mini/00482/image_12031.jpg,482,0,species_name_1,train
12045,data/raw/train_mini/00482/image_12045.jpg,482,0,species_name_1,train
```

| 字段 | 用途 |
|---|---|
| `image_id` | 全局唯一图片标识，**所有下游文件必须用这个id对齐**，不用文件名 |
| `image_path` | 图片相对路径 |
| `original_class_id` | iNaturalist原始类别编号 |
| `class_idx` | 统一编号 **0–499**，由`class_to_idx.json`确定 |
| `class_name` | 物种名称，方便展示和报告分析 |
| `split` | train / val / test |

### 3.2 类别映射文件（全组唯一标准）

⚠️ **命名与内容必须严格区分**，避免"名字叫class_to_idx实际存的是idx_to_class"这种误解：

**`class_to_idx.json`**（原始类别id → 统一索引，真正意义上的"class到idx"）

```json
{
  "482": 0,
  "3791": 1
}
```

**`idx_to_class.json`**（统一索引 → 详细类别信息，用于展示/报告）

```json
{
  "0": { "original_class_id": 482, "class_name": "Species A", "category": "Aves" },
  "1": { "original_class_id": 3791, "class_name": "Species B", "category": "Plantae" }
}
```

两个文件都要保存，`train/val/test.csv`生成时同时使用这两份映射做交叉验证，确保一致。

🔴 **B、C、D必须使用完全相同的 `class_idx = 0–499`**，不能出现"B的0号是鸟，C的0号是花"这种错位，否则E无法比较结果。

### 3.3 `selected_classes.csv`

```csv
class_idx,original_class_id,class_name,category
0,482,species_name_1,Aves
1,3791,species_name_2,Plantae
```

### 3.4 `split_config.json`

```json
{
  "random_seed": 9517,
  "num_classes": 500,
  "class_selection_method": "uniform_random_sampling",
  "train_images_per_class": 40,
  "val_images_per_class": 10,
  "test_images_per_class": 10,
  "image_size": 224,
  "dataset": "iNaturalist-2021"
}
```

> `class_selection_method` 字段说明500类的抽样方式（如完全随机 / 按大类比例抽样），方便报告Introduction/Dataset部分直接引用。

---

## 四、统一图像预处理接口

A提供两层接口：

### 4.1 原始图片接口（B使用，传统方法）

```python
dataset = INatDataset(
    csv_file="data/metadata/train.csv",
    transform=None
)
# 返回: image, class_idx, image_id
```

B可以只做resize，不做ImageNet normalisation。

### 4.2 CNN接口（C、D使用）

```python
train_loader = get_dataloader(
    split="train",
    transform_type="basic_aug",   # none / basic_aug / strong_aug
    batch_size=64
)
```

统一基础参数：

```text
Image size: 224 × 224
Colour mode: RGB
Normalisation: ImageNet mean/std
```

C的augmentation消融实验（No Aug / Basic Aug / Strong Aug）通过`transform_type`参数切换。

---

## 五、模型预测结果输出规范

B、C、D训练完成后，**必须**在测试集上跑推理，输出统一格式文件，E只读取这些文件，不重新运行任何人的模型代码。

### 5.1 `predictions.csv`

```csv
image_id,true_label,pred_label,top1_score,method_name,split
88031,0,0,0.8421,resnet18_scratch_basic_aug,test
88032,0,6,0.3512,resnet18_scratch_basic_aug,test
99120,1,1,2.7534,sift_bovw_svm,test
```

| 字段 | 含义 |
|---|---|
| `image_id` | **必须**与test.csv中的image_id一致，作为对齐key |
| `true_label` | 真实class_idx |
| `pred_label` | Top-1预测 |
| `top1_score` | 第一名的分类分数（注意：**不一定是概率**，SVM可能是decision score，字段名保留`top1_score`而非`probability`，避免误导） |
| `method_name` | 方法标识，见第六节命名清单 |
| `split` | 固定为`test`（正式评估只用test集） |

> `is_correct`列可由E在evaluation阶段自动生成，不作为必须提交字段。

### 5.2 `scores.npz`（分类分数矩阵，用于Top-5/排序分析）

⚠️ 不使用裸的`scores.npy`，因为它本身不包含`image_id`，无法验证与`predictions.csv`的对应关系。**统一改用`scores.npz`，把`image_id`和分数矩阵打包在一起，强制保证可对齐**：

```python
np.savez(
    "scores.npz",
    image_ids=image_ids,       # shape: [N]
    scores=scores,             # shape: [N, 500]
    class_indices=np.arange(500)
)
```

读取时：

```python
data = np.load("scores.npz")
image_ids = data["image_ids"]
scores = data["scores"]
```

**`scores[i, j]` 的含义**：第 `i` 张图片对第 `j` 类的**分类分数**（不强制是概率，可以是 decision score、logit 或 probability），分数越高代表模型认为该样本越可能属于该类，用于排序和计算Top-k。

各方法具体用什么分数，按各自最自然、最快的方式来，不强制统一成概率：

| 方法 | 使用的分数 |
|---|---|
| SVM | `decision_function()` 输出即可，**不强制`probability=True`**（该参数会触发额外的交叉验证概率校准，在20,000张图/500类的规模下可能显著拖慢训练，若B确认可以接受再开启） |
| Random Forest | `predict_proba()` |
| CNN（C/D） | softmax概率或logits均可 |

E计算Top-5时统一用：

```python
top5 = np.argsort(scores, axis=1)[:, -5:]
```

🔴 **重要**：由于不同方法的分数尺度不同（SVM的decision score、RF的probability、CNN的softmax不在同一量纲），**E不得跨模型直接比较"置信度大小"**，分数仅用于同一模型内部的类别排序（Top-k计算），跨模型比较只能用accuracy/F1等标准化指标。

🔴 **对齐规则**：`predictions.csv`与`scores.npz`均以`image_id`为准对齐，不假设行号天然一致；推理阶段一律设置 `shuffle=False`。

### 5.3 `runtime.json`

⚠️ **硬件约定按方法类型分开，不强制全部使用Colab T4**：SIFT/HOG/K-Means/SVM/Random Forest主要吃CPU，GPU对它们没有帮助，强行都用Colab T4意义不大，反而可能因为Colab分配的CPU负载不稳定而失真。改为：

> 同类型方法内部（传统方法之间、深度学习方法之间）的时间比较应尽量在相同硬件环境下完成；传统方法与深度学习方法之间的时间差异可作为实际运行成本参考，但**不做严格的跨类型效率结论**（因为CPU vs GPU本身不可比），需在报告Discussion中说明这一点。

传统方法（B）记录：

```json
{
  "training_time_seconds": 320.5,
  "inference_time_seconds": 8.2,
  "num_test_images": 5000,
  "hardware": {
    "platform": "Google Colab",
    "cpu": "Intel Xeon (Colab分配)",
    "gpu": null,
    "ram_gb": 12.7
  },
  "software": {
    "python": "3.11",
    "scikit_learn": "1.6.1"
  }
}
```

深度学习方法（C/D）记录：

```json
{
  "training_time_seconds": 8350.4,
  "inference_time_seconds": 94.2,
  "num_test_images": 5000,
  "hardware": {
    "platform": "Google Colab",
    "cpu": "Intel Xeon (Colab分配)",
    "gpu": "NVIDIA T4",
    "ram_gb": 12.7
  },
  "software": {
    "python": "3.11",
    "pytorch": "2.x"
  }
}
```

**C/D的正式对比实验（用于报告里training/inference time比较）统一在Colab T4上跑**，日常调试可用自己电脑，但最终报告引用的深度学习时间数据必须来自同一类硬件。若个别实验条件受限必须用本地机器，需如实注明，并在报告Discussion里说明硬件差异对时间对比的影响。

---

## 六、输出目录规范

```text
outputs/
├── traditional/
│   ├── sift_bovw_svm/
│   │   ├── predictions.csv
│   │   ├── scores.npz
│   │   ├── metrics.json
│   │   └── runtime.json
│   └── hog_random_forest/
│       ├── predictions.csv
│       ├── scores.npz
│       └── runtime.json
│
├── scratch/
│   ├── resnet18_scratch_no_aug/
│   ├── resnet18_scratch_basic_aug/
│   ├── resnet18_scratch_strong_aug/
│   ├── resnet18_scratch_longtail_unbalanced/    # A的长尾实验，挂在C的pipeline上跑
│   └── resnet18_scratch_longtail_resampled/     # 长尾+重采样对照组
│       ├── predictions.csv
│       ├── scores.npz
│       ├── training_history.csv
│       └── runtime.json
│
└── transfer/
    ├── resnet18_pretrained_frozen/
    └── resnet18_pretrained_finetuned/
        ├── predictions.csv
        ├── scores.npz
        ├── training_history.csv
        └── runtime.json
```

**method命名清单（唯一标准，与目录结构完全一致，不得自创新名字，需扩展先在群里同步）：**

```text
sift_bovw_svm
hog_random_forest
resnet18_scratch_no_aug
resnet18_scratch_basic_aug
resnet18_scratch_strong_aug
resnet18_scratch_longtail_unbalanced
resnet18_scratch_longtail_resampled
resnet18_pretrained_frozen
resnet18_pretrained_finetuned
```

⚠️ 统一用 `finetuned`（不用`finetune`），避免脚本、目录名、method_name三处出现不一致的拼写。

---

## 七、Evaluation Pipeline 调用方式

E提供统一脚本，B/C/D不需要自己写evaluation逻辑：

```bash
python src/evaluation/evaluate.py \
  --predictions outputs/traditional/sift_bovw_svm/predictions.csv \
  --scores outputs/traditional/sift_bovw_svm/scores.npz \
  --classes data/metadata/idx_to_class.json \
  --output outputs/traditional/sift_bovw_svm/evaluation
```

自动产出的评估指标（**完整清单，需在报告Experimental Results中全部体现**）：

```text
Top-1 accuracy
Top-5 accuracy
Overall accuracy      # 单标签分类下与Top-1 accuracy数值相同，报告中说明这一点即可
Macro precision
Macro recall
Macro F1
Mean per-class / balanced accuracy   （可选，建议加）
Confusion matrix（全类别 + 挑选子集的详细版）
Training time / Inference time（引用runtime.json）
```

对应输出文件：

```text
metrics.json
confusion_matrix.npy
confusion_matrix.png
per_class_metrics.csv
top_confused_pairs.csv
failure_cases.csv
```

---

## 八、Robustness 测试接口

**采用「实时退化」方式，不预先生成并存储退化图片文件**（避免4种退化 × 多severity × 5000张图造成的巨大存储和对齐成本）。

### 8.1 退化必须发生在统一的流程位置

🔴 **关键**：B、C、D的预处理方式不同（B可能转灰度/不做ImageNet normalisation，C/D做224×224+ImageNet normalisation），如果退化处理插入的位置不统一，会导致"实际测试的退化强度不公平"。**必须统一为以下顺序**：

```text
读取原始图片（PIL Image, RGB）
  → 应用退化（在原始RGB图上进行，退化前不做任何模型专属预处理）
  → 执行各模型自己的 resize / 特征提取 / normalisation
  → 推理
```

即：**退化永远发生在"读取原图之后、模型自己的预处理之前"**，不允许在CNN已经normalise之后再加噪声，也不允许B在灰度图上加噪声、C在RGB图上加噪声——退化操作的输入必须是统一的原始RGB图。

### 8.2 退化函数接口

```python
# src/evaluation/degradation.py
def apply_degradation(
    image: "PIL.Image.Image",
    degradation_type: str,
    severity: int,
    seed: int | None = None
) -> "PIL.Image.Image":
    """
    degradation_type: "gaussian_noise" / "blur" / "brightness" / "jpeg_compression"
    severity: 1-5 等级，具体参数由 configs/robustness.yaml 统一定义
    返回: 退化后的 PIL Image（RGB），交给各模型自己的预处理pipeline继续处理
    """
```

### 8.3 退化参数表（由E写入 `configs/robustness.yaml`，具体数值可后续微调）

| 退化类型 | Severity 1 | Severity 3 | Severity 5 |
|---|---:|---:|---:|
| Gaussian noise | σ=0.02 | σ=0.06 | σ=0.10 |
| Gaussian blur | radius=1 | radius=3 | radius=5 |
| Brightness | 0.9 | 0.7 | 0.5 |
| JPEG quality | 80 | 40 | 10 |

B/C/D在自己的推理脚本中调用`apply_degradation`函数，对同一批`test.csv`图片实时做退化处理后再推理，输出格式与第五节一致：

```bash
python predict.py \
  --test_csv data/metadata/test.csv \
  --degradation gaussian_noise \
  --severity 3 \
  --output outputs/robustness/resnet18_scratch_basic_aug/noise_3
```

**职责边界**：
- **E负责**：定义退化类型、severity等级、退化函数实现、Robustness测试流程设计
- **B/C/D负责**：保证自己的模型能调用E提供的`apply_degradation`函数处理输入图片，并输出统一格式结果；不得自行更改退化参数

---

## 九、长尾（Class Imbalance）实验的数据构造规则

基础训练集本身是**均衡的**（每类40张），因此`longtail`不能直接理解成"重采样后的模型"，必须包含完整的对照关系，否则无法证明"重采样是否真的有效"。

**构造步骤**：

1. 从基础训练集`train.csv`按预设规则构造**长尾分布**训练版本（例如保留少数类别较多样本、多数类别只保留5-10张，模拟真实长尾分布），生成 `longtail_train.csv`
2. 对长尾版本应用重采样或其他平衡策略（如过采样少数类、类别加权loss等），生成 `longtail_resampled_train.csv`
3. **Validation和Test集保持不变**，始终使用原始均衡的`val.csv`/`test.csv`，确保三组实验在同一把尺子下比较

**必须产出两组结果做对照**（而不是只有重采样后的单一结果）：

```text
resnet18_scratch_longtail_unbalanced    ← 长尾分布，未做任何平衡处理
resnet18_scratch_longtail_resampled     ← 长尾分布 + 重采样/加权
```

**需要保存的文件**：

- `longtail_train.csv` / `longtail_resampled_train.csv`
- `longtail_config.json`，记录长尾构造规则（如保留比例、衰减方式）与random seed，保证可复现：

```json
{
  "random_seed": 9517,
  "longtail_ratio": "exponential_decay",
  "min_images_per_class": 5,
  "max_images_per_class": 40,
  "resampling_strategy": "oversampling_minority"
}
```

报告里需要对比"均衡 vs 长尾未处理 vs 长尾+重采样"三组的整体accuracy/F1，以及**per-class表现**（尤其是样本量少的类别），才能说明重采样的实际效果。

---

## 十、Test Set 使用规则（防止Test Leakage）

🔴 严格区分Validation和Test的用途：

```text
Validation：用于模型选择、超参数调整、checkpoint选择、augmentation/重采样策略的效果验证。

Test：仅用于最终评价、error analysis、以及固定模型下的Robustness测试。
      不得用于选择模型、调整参数或决定任何训练策略。
```

具体禁止行为：
- 不允许"看了test结果发现不理想，回去调整模型/augmentation策略，再重新测test"，这种反复调参会造成test leakage，导致报告结果虚高、不可信
- Robustness测试固定使用已经训练/选定好的最终模型，退化测试的结果不能反过来用于修改模型
- 如果需要在开发过程中反复验证效果，一律使用validation集，只有在所有方法都最终定型后才统一跑一次test集评估

---

## 十一、全组数据合同（请所有人确认）

```text
1. 使用A生成的train.csv、val.csv、test.csv，任何成员不得自行重新划分数据。

2. 所有模型统一使用 class_idx 0–499，映射由 class_to_idx.json 确定，不得自行编号。

3. 每个模型的预测结果必须包含三个文件：
   predictions.csv
   scores.npz
   runtime.json

4. scores.npz 中 scores 维度必须为 [测试图片数量 × 500]，
   image_ids 维度为 [测试图片数量]，两者一一对应。

5. predictions.csv 与 scores.npz 均以 image_id 作为对齐依据，
   不假设两者行序天然一致；推理阶段一律 shuffle=False。

6. 所有正式实验使用同一份 test.csv（来自官方 iNat2021 validation set），
   且 test 集仅用于最终评估，不得用于模型选择或调参（见第十节）。

7. 所有模型的 Top-1/Top-5 accuracy、Macro Precision/Recall/F1、
   Confusion Matrix 等指标均由 E 提供的统一 Evaluation Pipeline 计算，
   个人不得自行计算后直接写入报告。

8. Robustness 的退化类型、severity 参数由 E 统一定义并提供函数，
   退化必须发生在"读取原图之后、各模型自己预处理之前"（见第八节），
   B/C/D 仅负责调用，不得自行更改退化参数或插入位置。

9. 时间对比按方法类型分开记录硬件环境（见5.3节），
   深度学习方法的正式对比实验统一使用 Colab T4 GPU 运行；
   若条件受限使用本地硬件，须在 runtime.json 中如实注明并在报告中说明。

10. method 名称严格使用第六节清单中的名称，如需新增，先在群内同步确认。

11. scores.npz 中的分数（decision score / probability / logit）
    仅用于同一模型内部的类别排序（Top-k计算），不得跨模型直接比较置信度大小。
```

---

## 十二、关键节点与负责人

| 内容 | 负责人 | 建议完成时间 |
|---|---|---|
| train/val/test.csv、class_to_idx.json、idx_to_class.json、split_config.json | A | Week 7 前半段 |
| Evaluation Pipeline 框架 + degradation.py + configs/robustness.yaml | E | Week 7 |
| B/C/D 确认能否按格式输出（尤其SVM decision_function是否够用） | B/C/D | Week 8 内反馈 |
| 长尾数据构造（longtail_train.csv等） | A（配合C跑实验） | Week 8-9 |
| 中期格式联调（各交一版初步predictions.csv试跑） | 全员 | Week 8 |

---

## 十三、给 B 的实操补充（传统方法特有的坑）

### 13.1 HOG特征维度可能过大，先小规模试跑

500类、20,000张训练图，如果HOG的cell size设置过细，单张图特征维度会很高，导致内存占用和训练时间失控。**建议先在50-100类的小子集上试跑**，记录以下几项数据，再决定正式参数：

```text
单张图特征维度
特征提取总耗时
内存占用峰值
Random Forest训练耗时
```

确认可行后再扩展到全部500类，避免正式跑到一半才发现内存不够或耗时过长。

### 13.2 BoVW的K-Means不要把所有SIFT descriptor一次性塞进去

20,000张训练图产生的SIFT descriptor总量可能非常庞大，直接用标准`KMeans`聚类会极慢甚至内存溢出。建议：

- 每张图最多随机采样固定数量的descriptor（例如每张图取100个），或者
- 从全体descriptor中随机采样一个总量上限（例如总共采样50万个descriptor）用于聚类
- 优先使用 `sklearn.cluster.MiniBatchKMeans` 而不是标准`KMeans`，速度会快很多

这两点建议在正式实验前先跑一次小规模测试确认可行，再应用到全量数据上。
