# 论文结构与 A–E 分工方案

> 文档状态：已确认的论文规划；已纳入完成的 full-data 扩展实验
> 目标格式：COMP9517 spec 指定的官方 CVPR LaTeX 模板，英文正文，提交 PDF
> 论文定位：在统一数据划分和评价协议下，对有限训练数据条件下的细粒度物种识别方法进行系统性实证比较
> 注意：本文不宣称提出全新的模型；所有定量结论必须来自最终实验输出和 E 的统一评价流程

## 课程 Spec 的格式与提交要求

以下要求来自 `COMP9517_26T2_Group_Project_Specification-v1.pdf`，优先于普通 CVPR 会议投稿规则和本组内部建议：

- 必须使用课程指定的官方 CVPR LaTeX 模板；
- 模板中必须使用 `\usepackage[pagenumbers]{cvpr}`；
- 只接受由该 LaTeX 模板渲染的 PDF；
- 正文最多 10 页，标题、Abstract、正文、diagrams、figures 和 tables 均计入正文页数；
- References 不计入 10 页限制，可以使用额外页面；
- 超过正文 10 页后的页面只能包含 References，不能放 Appendix、补充实验或额外图表；
- 最终报告 PDF 必须不超过 10 MB；
- 不遵守指定模板可能导致报告得零分。

内部排版目标为 8.5–9 页正文，为图表浮动、作者信息和最终修改预留空间；10 页是不可超过的硬性上限。

## 论文题目

**Fine-Grained Species Recognition with Limited Training Data: Augmentation, Transfer Learning, and Visual Explanations**

## 论文主线

本文研究在每类训练样本有限的 500 类细粒度物种识别任务中，表示学习方式、数据增强、模型容量和预训练如何影响分类性能与泛化能力。论文在相同的数据划分和评价协议下比较：

- handcrafted visual features；
- 从零训练的 ResNet18；
- ImageNet 预训练模型的 frozen 和 full fine-tuning 策略；
- regularization 和 self-attention 变体；
- 不同图像退化条件下的鲁棒性；
- 基于 Grad-CAM 的视觉解释和失败案例；
- 在固定 validation/test set 上，将训练集扩展到清洗后的 129,378 张图像后，随机初始化与 ImageNet 预训练的 ResNet18。

当前实验支持的核心结论是：在每类仅 40 张训练图的设置下，迁移学习明显优于从零训练；适度且保持类别语义的数据增强比过强的数据增强更有效；增加模型容量并不必然改善有限数据下的泛化。在 full-data 扩展中，fine-tuned ResNet18 仍以 81.58% 对 64.20% 的 test Top-1 优于同预算 scratch ResNet18，说明预训练优势在更大训练集上仍然存在，但该扩展应与 20,000 张图像的 limited-data 主实验分开报告。

## Research Questions

- **RQ1 — Representation:** How do handcrafted features, scratch-trained convolutional networks, and pretrained representations compare under limited training data?
- **RQ2 — Augmentation:** How does augmentation strength affect generalization in fine-grained species recognition?
- **RQ3 — Capacity:** Do regularization and additional model capacity improve transfer learning when only a small number of training images are available per class?
- **RQ4 — Reliability:** How do the evaluated models respond to image degradation, and what failure modes are revealed by visual explanations?
- **RQ5 — Data Scale:** Does the advantage of ImageNet initialization persist when the training set is expanded from 20,000 images to the cleaned 129,378-image full-data subset under a matched 15-epoch budget?

## 贡献列表草稿

1. We conduct a controlled comparison of handcrafted, scratch-trained, and pretrained representations using the same 500-class dataset split and evaluation protocol.
2. We evaluate how augmentation strength, class imbalance, regularization, and model capacity affect recognition when only 40 training images are available per class.
3. We combine quantitative evaluation, robustness testing, confusion analysis, and Grad-CAM visualizations to characterize model failures.
4. We provide a controlled full-data extension comparing scratch and ImageNet-initialized ResNet18 under the same 15-epoch budget, showing that fine-tuning retains a 17.38 percentage-point Top-1 advantage on the fixed test set.

以上贡献必须与最终 Experiments 中的表格或图像一一对应。正文避免使用 `novel`、`state-of-the-art` 或 `significant` 等缺乏证据的主观表述。

## 章节结构

### Abstract

**负责人：E 主写；A–D 核对各自方法和数字。**

建议控制在 180–220 词，最后完成。段落逻辑：

1. 介绍有限数据下的 500 类细粒度物种识别任务。
2. 说明类别间差异细微、类内变化大以及训练样本不足等挑战。
3. 概括 traditional、scratch 和 transfer learning 的统一比较。
4. 概括 augmentation、regularization、model capacity、robustness 和 Grad-CAM 分析。
5. 将 129,378-image full-data 实验明确写成独立扩展，而不是 limited-data 主实验的一部分。
6. 报告最重要的定量结果，包括 full-data Scratch 64.20% 与 Fine-tuned 81.58% test Top-1。
7. 总结预训练表示和容量—数据规模匹配的作用。

Abstract 不引用文献。所有结果句必须包含经过 E 的统一评价流程确认的真实数字。

### 1. Introduction

**负责人：A 完成初稿；E 根据全组结果统一改写和润色。**

建议写成五段：

1. **Task and motivation：**说明自动物种识别在生物多样性监测、公民科学和生态研究中的价值。
2. **Technical challenges：**介绍 iNaturalist 图像中的细粒度类别差异、背景变化、图像质量变化、有限样本和类别不平衡问题。
3. **Methodological motivation：**解释为什么需要在相同条件下比较 handcrafted features、scratch CNN 和 transfer learning。
4. **Study overview and findings：**概括数据划分、方法族、统一评价、full-data 扩展和主要发现。
5. **Contributions：**列出四条可由实验验证的贡献。

本节应在末尾自然引出 RQ1–RQ5，但不展开实现细节。

### 2. Literature Review

**负责人：B 负责整节合并；C 和 D 提供各自方向的参考文献与技术要点。**

#### 2.1 Handcrafted Visual Representations

**负责人：B。**

内容：

- Scale-Invariant Feature Transform（SIFT）；
- Bag-of-Visual-Words（BoVW）；
- Histogram of Oriented Gradients（HOG）；
- Support Vector Machine（SVM）和 Random Forest；
- handcrafted features 在大规模细粒度识别中的能力和局限。

#### 2.2 Deep Fine-Grained Recognition

**负责人：B 主写；C 提供 scratch CNN 和 augmentation 内容。**

内容：

- convolutional neural networks；
- ResNet 和 residual learning；
- 从零训练对数据规模的依赖；
- data augmentation 对泛化的影响；
- 过强增强可能破坏颜色、纹理和局部形态等类别判别线索。

#### 2.3 Transfer Learning and Visual Explanations

**负责人：D 提供内容；B 负责统一语气。**

内容：

- ImageNet pretraining；
- frozen feature extractor 和 full fine-tuning；
- limited-data settings 中的 transfer learning；
- Gradient-weighted Class Activation Mapping（Grad-CAM）；
- visual explanations 在 failure analysis 中的用途和局限。

Literature Review 应突出本文的研究空白是“统一条件下的系统比较和错误分析”，而不是声称以往没有任何相关方法。

### 3. Methods

#### 3.1 Dataset Construction

**负责人：A。**

需要说明：

- 数据来源为 iNaturalist 2021；
- 使用固定的 500 类子集和统一的 `class_idx = 0,\ldots,499`；
- 当前 mini 设置包含 20,000 张训练图、5,000 张 validation 图和 5,000 张 test 图；
- 每类分别包含 40/10/10 张 train/validation/test 图；
- full-data 扩展保持相同 500 类映射以及同一组 5,000 张 validation、5,000 张 test 图，只把训练集替换为质量扫描后的 `train_full_clean.csv`，共 129,378 张图；
- dataset construction seed 为 500；
- validation 用于模型选择和超参数调整；
- test 仅用于最终评价、固定模型的鲁棒性测试和错误分析；
- 长尾数据的构造规则以及 weighted sampling 设置；
- 数据质量扫描、重复图像处理和数据泄漏防护。

拟使用：

- **Table 1 — Dataset Statistics**：分行报告 limited-data（20,000 train）和 full-data extension（129,378 train），二者共享 5,000 validation 与 5,000 test。

正文必须将 full-data 明确标记为扩展实验；不得用 129,378 张训练图描述基于 20,000 张 mini 训练集得到的主结果，也不得将质量扫描前的 129,549 张候选图误写成最终训练数量。

#### 3.2 Handcrafted Baselines

**负责人：B。**

需要说明：

- SIFT + BoVW + Linear SVM；
- HOG + Random Forest；
- 图像 resize、灰度处理和特征标准化；
- SIFT descriptor sampling；
- MiniBatchKMeans vocabulary construction；
- BoVW vocabulary size、SVM 参数、HOG 参数和 Random Forest 参数；
- 模型训练、推理和输出格式。

#### 3.3 Scratch-Trained ResNet18

**负责人：C。**

需要说明：

- ResNet18 使用 `weights=None`；
- 输入大小为 \(224 \times 224\)，输出为 500 类 logits；
- Cross-Entropy Loss；
- AdamW/SGD、学习率、weight decay 和 cosine scheduler；
- best validation accuracy checkpoint selection；
- no augmentation、basic augmentation 和 strong augmentation；
- long-tail unbalanced 与 weighted sampling 设置；
- training seed 为 9517。

#### 3.4 Transfer Learning

**负责人：D。**

需要说明：

- ImageNet-pretrained ResNet18；
- frozen strategy：只训练最终分类层；
- fine-tuned strategy：更新全部参数；
- 两种策略的学习率、优化器、scheduler 和 checkpoint 规则；
- regularized variant：dropout、weight decay 和 label smoothing；
- self-attention variant：在 `layer4` 后加入 multi-head self-attention。

Self-attention 应明确写成用于分析模型容量的实验变体，不应表述为本文提出的主要新模型。

#### 3.5 Full-Data Scratch vs Fine-Tuned Extension

**负责人：A 说明数据扩展与质量控制；C 负责 Scratch；D 负责 Fine-tuned；E 负责公平性核对与统一评价。**

需要说明：

- 使用清洗后的 129,378-image `train_full_clean.csv`，validation 和 test 与 limited-data 主实验完全相同；
- 两种方法均使用 ResNet18、basic augmentation、15 epochs、batch size 64、8 workers、AdamW、weight decay \(10^{-4}\)、cosine schedule、AMP 和 seed 9517；
- Scratch 使用随机初始化和 \(10^{-3}\) learning rate；
- Fine-tuned 使用 ImageNet initialization、全参数更新和 \(10^{-4}\) learning rate；
- 两种方法都按 best validation accuracy 选择 checkpoint；
- 两个训练作业均使用 NVIDIA L40S、相同软件版本和相同 PBS 资源请求，但运行在不同节点，因此训练时间属于同 GPU 型号下的近似可比结果，不应描述为完全相同硬件；
- 该实验只控制 full-data 条件下的初始化方式，不把它与 mini 中训练轮数和优化器不同的最佳 Scratch 模型解释为纯粹的数据规模消融。

#### 3.6 Evaluation and Analysis Protocol

**负责人：E 主写评价协议；D 补充 Grad-CAM 方法。**

需要说明：

- Top-1 accuracy；
- Top-5 accuracy；
- Macro Precision、Macro Recall 和 Macro F1；
- Balanced Accuracy；
- training time 和 inference time；
- confusion matrix、per-class metrics 和 confusable pairs；
- Gaussian noise、blur、brightness reduction 和 JPEG compression；
- 每种退化的五个 severity levels；
- Grad-CAM 的 target layer、梯度加权和可视化方式。

所有模型必须使用相同的 test set。退化必须发生在读取原始 RGB 图像之后、模型专属预处理之前。

### 4. Experimental Results

#### 4.1 Experimental Setup

**负责人：E。**

统一说明：

- train/validation/test 划分；
- Katana/V100 或实际使用的硬件环境；
- 图像大小、batch size 和随机种子；
- 各方法的训练轮数和 checkpoint selection rule；
- B/C/D 的正式 runtime 必须来自可比的运行环境；
- limited-data 主实验和 full-data 扩展必须在表格与正文中分组，不能把不同训练集规模的方法放在同一排名中；
- full-data Scratch 与 Fine-tuned 均使用 NVIDIA L40S、PyTorch 2.3.1 + CUDA 12.1、15 epochs 和相同 PBS 资源请求；分别记录节点 `k098` 与 `k097`；
- 所有指标由 E 的统一 evaluation pipeline 计算。

#### 4.2 Main Comparison

**负责人：E；B/C/D 核对自己的原始输出。**

比较：

- `sift_bovw_svm`；
- `hog_random_forest`；
- `resnet18_scratch_basic_aug_sgd` 或最终选定的 scratch model；
- `resnet18_pretrained_frozen`；
- `resnet18_pretrained_finetuned`。

拟使用：

- **Table 2 — Overall Performance Comparison**
- **Fig. 1 — Accuracy versus Training/Inference Time**

正文首先回答 RQ1，然后分析性能、计算成本和数据效率之间的权衡。

#### 4.3 Effect of Data Augmentation

**负责人：C。**

当前可用结果：

| Setting                                  |  Top-1 |  Top-5 | Macro F1 |
| ---------------------------------------- | -----: | -----: | -------: |
| No augmentation                          | 22.62% | 44.38% |   22.52% |
| Basic augmentation                       | 31.76% | 57.42% |   31.44% |
| Strong augmentation                      | 25.88% | 51.00% |   24.52% |
| Basic augmentation + longer SGD training | 37.94% | 61.24% |   37.51% |

重点分析：

- basic augmentation 相比 no augmentation 提升 9.14 个百分点 Top-1；
- strong augmentation 优于 no augmentation，但低于 basic augmentation；
- 颜色、纹理和局部形态对物种识别重要，因此 augmentation 并非越强越好；
- longer training 和 optimizer choice 对 scratch baseline 的影响。

拟使用：

- **Fig. 2 — Scratch ResNet18 Training Curves**
- **Table 3 — Augmentation and Training Ablation**

#### 4.4 Transfer Learning and Capacity Ablation

**负责人：D。**

当前可用结果：

| Variant        |  Top-1 |      Top-5 | Macro F1 | Train/Val Gap |
| -------------- | -----: | ---------: | -------: | ------------: |
| Frozen         | 53.88% |     76.68% |   53.55% |    待最终统一 |
| Fine-tuned     | 66.24% |     85.82% |   66.20% |      31.97 pp |
| Regularized    | 65.20% |     85.42% |   65.11% |      24.43 pp |
| Self-attention | 62.08% |     82.82% |   62.04% |      36.18 pp |

重点分析：

- fine-tuned ResNet18 比当前最佳 scratch model 高 28.30 个百分点 Top-1；
- fine-tuning 比 frozen strategy 高 12.36 个百分点 Top-1；
- regularization 将 train/validation gap 缩小约 7.54 个百分点，但只损失约 1.04 个百分点 Top-1；
- self-attention 增加了未预训练参数和模型容量，但没有改善泛化；
- limited-data setting 中模型容量需要与数据规模匹配。

未经过 E 统一评价确认的指标继续标记为“待最终统一”，不得自行补数。

#### 4.5 Full-Data Scaling Extension

**负责人：C 提交 Full Scratch 结果；D 提交 Full Fine-tuned 结果；E 负责统一表格、差值计算和论述；A 核对 full-data 数量与质量扫描。**

统一评价得到：

| Full-data method                   | Test Top-1 | Test Top-5 | Macro F1 | Best Val Acc. | Best Epoch | Training Time |
| ---------------------------------- | ---------: | ---------: | -------: | ------------: | ---------: | ------------: |
| ResNet18 Scratch                   |     64.20% |     84.76% |   64.02% |        65.00% |         15 | 1,295.0 s |
| ResNet18 ImageNet Fine-tuned       | **81.58%** | **93.70%** | **81.53%** |    **81.44%** |         13 | 1,351.7 s |

关键结果与写作边界：

- 在相同 full-data、模型架构、15-epoch 预算和 L40S GPU 型号下，Fine-tuned 比 Scratch 高 17.38 个百分点 Top-1、8.94 个百分点 Top-5 和 17.52 个百分点 Macro F1；
- Fine-tuned 的训练时间比 Scratch 多 56.7 秒，约增加 4.38%；由于作业运行在不同 L40S 节点，该差异只能作为近似成本比较；
- Full Fine-tuned 的 81.44% 是 best validation accuracy，最终 test Top-1 是 81.58%，二者不得混写；
- limited-data 中当前最佳 Scratch 使用 100 epochs + SGD，而 full-data Scratch 使用 15 epochs + AdamW，因此跨数据规模的 37.94%→64.20% 只能作描述性观察，不能把 26.26 个百分点全部归因于新增训练图；
- limited-data Fine-tuned 与 full-data Fine-tuned 都使用 15-epoch AdamW 配置，test Top-1 从 66.24% 增至 81.58%；仍需把它表述为数据规模扩展下的观察结果，而不是多随机种子的因果估计。

证据文件：

- `outputs/full_train/resnet18_scratch_full_basic_aug/evaluation/metrics.json`
- `outputs/full_train/resnet18_pretrained_finetuned/evaluation/metrics.json`
- `outputs/full_train/*/runtime.json`
- `outputs/full_train/*/training_history.csv`

拟使用：

- **Table 5 — Full-Data Scratch versus Fine-Tuned ResNet18**
- **Fig. 3 — Full-Data Scratch and Fine-Tuned Training Curves**

#### 4.6 Class Imbalance and Robustness

##### 4.6.1 Class Imbalance

**负责人：A 主写数据构造和分析；C 提供模型训练结果。**

比较：

- balanced training；
- long-tail unbalanced；
- long-tail with weighted sampling。

除整体 accuracy 和 Macro F1 外，还需要分析 minority classes 的 per-class recall/F1，以及样本数量与类别表现之间的关系。

##### 4.6.2 Robustness

**负责人：E。**

比较四种退化和五个 severity levels，回答：

- 哪种退化对各方法伤害最大；
- traditional、scratch 和 transfer 方法的性能下降速度是否不同；
- clean accuracy 较高是否同时意味着更强的鲁棒性。

拟使用：

- **Fig. 4 — Robustness under Image Degradations**

正式 robustness 结果已经生成；Abstract、Discussion 和 Conclusion 中的相关结论均应以
`report/generated/robustness_summary.csv` 为准。

#### 4.7 Visual Explanations and Failure Analysis

**负责人：D。**

使用 Grad-CAM 分析：

1. frozen 与 fine-tuned 模型的注意区域差异；
2. 数据质量或拍摄问题；
3. attention 正确但分类错误的真实细粒度混淆；
4. 多目标干扰；
5. 同属物种的系统性混淆；
6. 分类错误与 taxonomic relatedness 是否一致。

拟使用：

- **Fig. 5 — Grad-CAM and Representative Failure Cases**

正文需要把热力图位置、模型预测和错误类型联系起来，避免仅使用“注意力更集中”作为结论。

### 5. Discussion

**负责人：E 主写；C/D 提供 augmentation、capacity 和 Grad-CAM 证据。**

#### 5.1 Interpretation of Results

围绕以下问题组织：

- 为什么 ImageNet pretraining 在每类仅 40 张图时具有明显优势；
- 为什么保持类别语义的 augmentation 比过强 augmentation 更可靠；
- 为什么随机初始化的额外模型容量可能加剧过拟合；
- 为什么在训练集扩展到 129,378 张图后预训练仍有 17.38 个百分点 Top-1 优势，以及该优势相较 limited-data 观察差距缩小意味着什么；
- clean performance、robustness 和 explanation 是否给出一致结论；
- handcrafted、scratch 和 transfer 方法之间的性能—成本权衡。

#### 5.2 Limitations

必须说明的局限：

- 研究使用一个 500 类 iNaturalist 子集；
- 当前主要结果来自每类 40 张训练图；
- 深度学习主干主要限于 ResNet18；
- Grad-CAM 是定性解释，不能作为因果证明；
- 部分实验可能只有一个训练随机种子；
- full-data 扩展只比较一个 seed、一个 15-epoch 预算和两种初始化方式，没有覆盖 frozen、regularized、attention 或多种 optimizer；
- Full Scratch 与 Full Fine-tuned 使用相同 L40S 型号但不同物理节点，因此约 4.38% 的训练时间差不能解释为严格的算法加速或减速；
- 跨 limited-data/full-data 比较中，当前最佳 Scratch 的训练轮数和 optimizer 不完全匹配，因此不能把跨规模提升完全归因于数据数量。

### 6. Conclusion

**负责人：E 主写；A 审阅。**

建议使用一个简洁段落：

1. 重述研究问题；
2. 总结最重要的定量证据；
3. 总结 full-data 扩展中 Fine-tuned 81.58% 对 Scratch 64.20% 的结果，并明确它补充而不替代 limited-data 主实验；
4. 总结 augmentation、transfer learning 和 visual explanations 的联合意义；
5. 提出多随机种子、更强 fine-grained architecture、更多 full-data ablations 或 taxonomic supervision 等未来方向。

### References

**全员负责添加自己章节使用的真实文献；E 负责去重、统一 BibTeX key 和格式。**

每条参考文献需要能够在原始论文、官方 proceedings 或出版社页面中核验。任何新增引用都应记录其对应的正文 claim。

## 图表规划

| 顺序    | 图表                                      | 位置        | 负责人 | 核心作用                                   |
| ------- | ----------------------------------------- | ----------- | ------ | ------------------------------------------ |
| Table 1 | Dataset Statistics                        | Section 3.1 | A      | 说明 500 类及 train/validation/test 规模   |
| Fig. 1  | Accuracy versus Training/Inference Time   | Section 4.2 | E      | 展示主要方法的性能—成本权衡                |
| Table 2 | Overall Performance Comparison            | Section 4.2 | E      | 汇总 traditional、scratch 和 transfer 指标 |
| Fig. 2  | Scratch ResNet18 Training Curves          | Section 4.3 | C      | 展示不同 augmentation 的收敛和过拟合       |
| Table 3 | Augmentation and Training Ablation        | Section 4.3 | C      | 定量回答 RQ2                               |
| Table 4 | Transfer Learning and Capacity Ablation   | Section 4.4 | D      | 定量回答 RQ3                               |
| Table 5 | Full-Data Scratch versus Fine-Tuned       | Section 4.5 | C/D/E  | 在同一 full-data 预算下回答 RQ5            |
| Fig. 3  | Full-Data Scratch/Fine-tuned Curves       | Section 4.5 | C/D    | 展示 15 epochs 的收敛与泛化差异            |
| Fig. 4  | Robustness under Image Degradations       | Section 4.6 | E      | 比较四类退化和五档 severity                |
| Fig. 5  | Grad-CAM and Representative Failure Cases | Section 4.7 | D      | 展示注意区域和代表性失败模式               |

表格使用 `booktabs`，不使用竖线，最优结果加粗，次优结果加下划线。所有图片优先保存为 PDF/SVG 矢量格式，并同时保留 300 dpi PNG 预览。

## A–E 最终分工

| 成员 | 主要写作责任                                                                              | 实验与图表责任                                   | 交付给谁 |
| ---- | ----------------------------------------------------------------------------------------- | ------------------------------------------------ | -------- |
| A    | Introduction 初稿、Dataset、Class Imbalance、Conclusion 审阅                              | dataset statistics、full-data 清洗后数量核对、long-tail 数据与结果 | E        |
| B    | Literature Review 合并、Handcrafted Baselines、传统方法结果分析                           | SIFT/HOG 参数、预测输出和传统方法结果            | E        |
| C    | Scratch ResNet18、Augmentation、长尾模型、Full-data Scratch 部分                           | augmentation 表格、full-data training curves、scratch 输出 | E        |
| D    | Transfer Learning、capacity ablation、Full-data Fine-tuned、Grad-CAM failure analysis       | transfer/full-data 表格、Grad-CAM 图和 transfer 输出 | E        |
| E    | Abstract、Evaluation Protocol、Main Results、Full-data 对比、Robustness、Discussion、Conclusion、全文整合 | 主结果表、full-data 差值与公平性核对、性能—耗时图、鲁棒性图、统一指标 | 全组核对 |

## 协作与审阅关系

- A 的数据统计由 C 和 E 交叉检查，保证代码、CSV 和论文一致。
- B 的传统方法描述由 E 检查输出合同和指标可比性。
- C 的 augmentation 数字由 E 使用统一 pipeline 复算。
- D 的 transfer 和 Grad-CAM 结论由 C/E 检查是否存在过度推断。
- Full-data 的 129,378 张训练图统计由 A/E 核对，C/D 分别核对 Scratch/Fine-tuned 配置与输出，E 复算 17.38 pp Top-1 差值并确认两次训练均为 L40S。
- E 完成全文整合后，A–D 各自核对本方法的参数、数字、图表和引用。

## 推荐交付顺序

1. A 冻结数据统计、类别映射和划分说明。
2. B/C/D 分别提交方法描述、正式预测文件、runtime 和本方法结果分析。
3. E 使用统一评价代码生成所有方法的主结果表。
4. C 完成 augmentation 表格和训练曲线。
5. D 完成 transfer ablation 和 Grad-CAM 图。
6. C/D 提交两个 full-data 输出，E 完成 Table 5、Fig. 3 和 RQ5 分析。
7. E 完成 robustness、Discussion 和 Conclusion。
8. E 根据所有最终结果撰写 Abstract，并统一 Introduction。
9. A–D 完成数字核对，E 统一术语、时态、图表编号和参考文献。

## 每位成员的交付清单

每位成员应向 E 提交：

- 对应章节的 `.tex` 片段；
- 正式实验配置和方法参数；
- `predictions.csv`、`scores.npz` 和 `runtime.json`；
- full-data 负责人还需提交 `training_history.csv`、best/last checkpoint 的保存位置以及 PBS job/GPU 记录；
- 图表 PDF/SVG、PNG 预览和生成图表所用的原始数据；
- 本章节使用的 BibTeX 条目；
- 一份简短的 claim–evidence 对照：每个定量或定性结论对应哪个结果文件、表格或图像。

## 篇幅建议

课程 spec 的硬性上限为 10 页正文，且所有 diagrams、figures 和 tables 均计入页数。建议将内部目标控制在 8.5–9 页，不要以“刚好 10 页”为初稿目标。

| 内容                       |                           建议篇幅 |
| -------------------------- | ---------------------------------: |
| Title、authors 和 Abstract | 0.5–0.7 页；Abstract 约 180–220 词 |
| Introduction               |                         0.8–1.0 页 |
| Literature Review          |                         0.5–0.7 页 |
| Methods（包括 Dataset）    |                         2.0–2.3 页 |
| Experimental Results       |                         3.5–4.0 页 |
| Discussion                 |                         0.6–0.8 页 |
| Conclusion                 |                         0.2–0.3 页 |

References 不计入上述正文页数，可以从正文之后继续使用额外页面，但额外页面只能包含参考文献。最终 PDF 必须控制在 10 MB 以内。

如果篇幅不足，优先压缩 Literature Review、implementation details 和重复图表，不应删除主结果、关键消融、实验公平性说明或局限性。不得通过缩小字体、边距或修改 CVPR 模板来塞入内容。

## 当前结果状态

已在仓库文档中记录：

- Scratch ResNet18 augmentation 和 optimizer 结果；
- frozen、fine-tuned、regularized 和 self-attention transfer learning 结果；
- full-data Scratch 与 Fine-tuned 的训练、预测和统一评价结果：64.20% 对 81.58% test Top-1，均使用 NVIDIA L40S；
- full-data 的 Top-5、Macro F1、training time、training curves、confusion matrices 和 per-class/failure-case 输出；
- 部分 Grad-CAM failure analysis；
- dataset split 和数据质量扫描结果。

E 的统一评价流程现已补齐并核验：

- SIFT-BoVW-SVM、HOG-Random Forest、最佳 Scratch、Frozen 和 Fine-tuned
  五个 limited-data 方法的正式 Top-1、Top-5、Macro-F1 与 runtime；
- 四个 Scratch 设置、四个 Transfer 设置以及两个 full-data 设置的统一表；
- 12 个运行都包含相同的 5,000 个 test `image_id -> true_label` 映射；
- `report/generated/` 中的主结果表、消融表、full-data 表和两张可复现图。

仍需最终补齐：

- long-tail unbalanced 与 weighted sampling 对照结果；
- Grad-CAM 最终排版图和 A--D 对各自方法参数、数字及文字的复核。

E 的真实模型 robustness 已完成：4 个模型、4 类退化、5 档 severity，共
80 个完整运行；同环境 severity-0 基线、汇总 CSV 和论文 PDF/PNG 图均已生成。

limited-data runtime 来自不同硬件，已保留为描述性成本记录，不再要求把现有模型
迁移到同一环境重训。正式 robustness 运行已在同一本地环境中加载固定 checkpoint；
但不同模型的预处理计算量仍不同。在尚未完成的结果处继续保留明确待填标记，禁止
估计或编造数字。
