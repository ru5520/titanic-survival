# Titanic Survival Prediction

## 功能
- 训练随机森林 / XGBoost 模型并对比结果
- 生成可视化分析图表（特征重要性、存活率分析、雷达图、模型对比）
- 输入乘客信息，实时预测存活概率
- 输出预测原因分析（有利/不利因素逐条说明）
- 生成乘客个人雷达图

## 业务场景
模拟真实风险人群识别场景：根据乘客的个人信息（年龄、性别、
船票等级等），预测其在灾难中的存活概率。
这类二分类模型在金融风控、医疗预警、保险定价等业务中广泛应用。

## 技术方案
- 数据清洗：缺失值填补（中位数/众数）、字符串列数值化
- 特征工程：
  - 从姓名提取称谓（Title）
  - 构造家庭人数（FamilySize）
  - Pclass × Sex 交互特征
  - 年龄分段（AgeBand）
  - 甲板信息提取（Deck）
- 模型：随机森林 / XGBoost 对比实验
- 评估：5折交叉验证 + Kaggle Public Score

## 实验结果
| 模型 | 交叉验证均值 | Kaggle分数 |
|------|------------|-----------|
| 随机森林 baseline | 0.8238 | 0.77990 |
| 随机森林 + 新特征 | 0.8238 | 0.78229 |
| XGBoost 调参版 | 0.8339 | 0.76794 |

最终采用随机森林 + 特征工程版本，Kaggle Public Score：0.78229

## 核心结论
- Sex 和 Title 是最强预测特征，合计贡献超过50%的特征重要性
- 数据量小（891条）时随机森林泛化能力优于XGBoost
- 交互特征（Pclass_Sex）对一等舱误判问题有改善效果

## 技术栈
Python / pandas / scikit-learn / XGBoost

## 文件结构


## 运行方式
```bash
pip install pandas scikit-learn xgboost matplotlib
python main.py