import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import cross_val_score

def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def engineer_features(df):
    # 从 Name 里提取称谓
    df['Title'] = df['Name'].str.extract(r",\s([A-Za-z]+)\.")
    rare_titles = df["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 10].index 
    df["Title"] = df["Title"].apply(lambda x: x if x not in rare_titles else x)
    
     #有无 Cabin 记录
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    
    #  家庭人数
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    #独自一人
    df["IsAlone"] = (df["FamilySize"]==1).astype(int)

    #填补缺失项
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    # 把字符串转换成数字
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})  
    title_map ={"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    df["Title"] = df["Title"].map(title_map).fillna(4)

    # 交互特征
    df["Pclass_Sex"] = df["Pclass"] * 10 + df["Sex"]

    # 年龄分段
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]).astype(float)

    df["Deck"] = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else "U")
    deck_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7, "U": 8}
    df["Deck"] = df["Deck"].map(deck_map)

    return df

def select_features(df):
    features = ["Pclass", "Sex", "Age", "Fare", "Embarked",
                "HasCabin", "FamilySize", "IsAlone", "Title",
                "Pclass_Sex", "AgeBand", "Deck"]
    return df[features]

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200,max_depth=7,random_state=42)
    model.fit(X_train,y_train)
    return model


def evaluate_model(model,X_val,y_val,features_names):
    predictions = model.predict(X_val)

    print(f"准确率：{accuracy_score(y_val,predictions):4f}")
    print()
    print("分类报告: ")
    print(classification_report(y_val,predictions,target_names=["死亡","存活"]))

    print("======== 特征重要性 ========")
    importances=pd.Series(model.feature_importances_,index=features_names)
    print(importances.sort_values(ascending=False))

def cross_validate_model(X,y):
    model = RandomForestClassifier(n_estimators=200,max_depth=7,random_state=42)
    scores = cross_val_score(model,X,y,cv=5,scoring="accuracy")
    print(f"每折的准确率: {scores.round(4)}")
    print(f"平均准确率: {scores.mean().mean():.4f}")
    print(f"标准差: {scores.std():.4f}")

def get_feature_importance(model,X_text,text_df):
        predictions = model.predict(X_text)
        submission = pd.DataFrame({"PassengerId":text_df["PassengerId"],"Survived":predictions})
        submission.to_csv("submission.csv",index=False)
        print(f"提交文件已保存为 submission.csv")
        
def generate_submission(model, X_test, test_df):
    predictions = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv("submission.csv", index=False)
    print("\n提交文件已生成：submission.csv")

def error_analysis(model, X_val, y_val, train_df, val_index):
    predictions = model.predict(X_val)

    val_original = train_df.iloc[val_index].copy()
    val_original["Predicted"] = predictions
    val_original["Actual"] = y_val.values
    val_original["Correct"] = (predictions == y_val.values)

    errors = val_original[val_original["Correct"] == False]

    print("===== 误差分析 =====")
    print(f"总错误数：{len(errors)}")

    print("\n错误样本的性别分布：")
    print(errors["Sex"].value_counts())

    print("\n错误样本的船票等级分布：")
    print(errors["Pclass"].value_counts())

    print("\n错误样本的年龄统计：")
    print(errors["Age"].describe())

def train_xgb(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def plot_feature_importance(model, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="steelblue")
    plt.title("特征重要性")
    plt.xlabel("重要性得分")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("已保存：feature_importance.png")

def plot_survival_rate(train_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 性别存活率
    train_df.groupby("Sex")["Survived"].mean().plot(
        kind="bar", ax=axes[0], color=["steelblue", "salmon"]
    )
    axes[0].set_title("性别存活率")
    axes[0].set_xticklabels(["男", "女"], rotation=0)
    axes[0].set_ylabel("存活率")

    # 船票等级存活率
    train_df.groupby("Pclass")["Survived"].mean().plot(
        kind="bar", ax=axes[1], color=["gold", "silver", "#cd7f32"]
    )
    axes[1].set_title("船票等级存活率")
    axes[1].set_xticklabels(["一等舱", "二等舱", "三等舱"], rotation=0)

    # 年龄分布
    train_df[train_df["Survived"] == 1]["Age"].plot(
        kind="kde", ax=axes[2], label="存活", color="green"
    )
    train_df[train_df["Survived"] == 0]["Age"].plot(
        kind="kde", ax=axes[2], label="死亡", color="red"
    )
    axes[2].set_title("年龄分布对比")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("survival_analysis.png", dpi=150)
    plt.close()
    print("已保存：survival_analysis.png")

def plot_radar(model, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance / importance.sum()

    labels = importance.index.tolist()
    values = importance.values.tolist()
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="steelblue", linewidth=2)
    ax.fill(angles, values, color="steelblue", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    ax.set_title("特征重要性雷达图", fontsize=14, pad=20)
    ax.set_rlabel_position(30)

    plt.tight_layout()
    plt.savefig("feature_radar.png", dpi=150)
    plt.close()
    print("已保存：feature_radar.png")

def plot_model_comparison():
    models = ["RF Baseline", "RF + 新特征", "XGBoost"]
    cv_scores = [0.8238, 0.8238, 0.8339]
    kaggle_scores = [0.77990, 0.78229, 0.76794]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, cv_scores, width, label="交叉验证均值", color="steelblue")
    ax.bar(x + width/2, kaggle_scores, width, label="Kaggle分数", color="salmon")

    ax.set_ylabel("准确率")
    ax.set_title("模型对比：交叉验证 vs Kaggle分数")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0.7, 0.9)

    for i, (cv, kg) in enumerate(zip(cv_scores, kaggle_scores)):
        ax.text(i - width/2, cv + 0.003, f"{cv:.4f}", ha="center", fontsize=9)
        ax.text(i + width/2, kg + 0.003, f"{kg:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()
    print("已保存：model_comparison.png")

def analyze_reasons(name, passenger, survive_prob):
    print("\n===== 预测原因分析 =====")

    reasons_negative = []
    reasons_positive = []

    if passenger["Sex"] == 1:
        reasons_positive.append("女性乘客，Titanic 执行女性优先救援政策，存活优势极大")
    else:
        reasons_negative.append("男性乘客，历史数据显示男性存活率仅约 19%，处于明显劣势")

    if passenger["Pclass"] == 1:
        reasons_positive.append("一等舱乘客，距离救生艇最近，优先疏散，存活率约 63%")
    elif passenger["Pclass"] == 2:
        reasons_positive.append("二等舱乘客，存活率约 47%，处于中等水平")
    else:
        reasons_negative.append("三等舱乘客，位于船舱底层，疏散路径最长，存活率仅约 24%")

    if passenger["AgeBand"] == 0:
        reasons_positive.append("儿童（12岁以下），救援中优先照顾儿童，存活概率较高")
    elif passenger["AgeBand"] == 1:
        reasons_positive.append("青少年（13-18岁），年龄较小，有一定救援优先级")
    elif passenger["AgeBand"] == 2:
        reasons_positive.append("青壮年（19-35岁），体力较好，自救能力强")
    elif passenger["AgeBand"] == 3:
        reasons_negative.append("中年（36-60岁），体力开始下降，自救能力有所减弱")
    else:
        reasons_negative.append("老年（60岁以上），行动不便，自救能力最弱")

    if passenger["Fare"] >= 100:
        reasons_positive.append(f"票价 {passenger['Fare']:.1f} 英镑，属于高额票价，通常对应更好的舱位和位置")
    elif passenger["Fare"] >= 30:
        reasons_positive.append(f"票价 {passenger['Fare']:.1f} 英镑，属于中等票价")
    else:
        reasons_negative.append(f"票价 {passenger['Fare']:.1f} 英镑，票价偏低，通常对应较差的舱位位置")

    if passenger["HasCabin"] == 1:
        reasons_positive.append("有舱位记录，通常为头等舱或二等舱乘客，距救生艇较近")
    else:
        reasons_negative.append("无舱位记录，多为三等舱或统舱乘客，疏散条件较差")

    family = passenger["FamilySize"]
    if family == 1:
        reasons_negative.append("独自一人乘船，无家人协助，紧急情况下互助能力为零")
    elif family <= 4:
        reasons_positive.append(f"与 {family-1} 位家人同行，适中的家庭规模，互相照应有利于存活")
    else:
        reasons_negative.append(f"家庭人数达 {family} 人，人数过多反而难以统一疏散，存活率更低")

    print("\n有利因素：")
    if reasons_positive:
        for i, r in enumerate(reasons_positive, 1):
            print(f"  {i}. ✓ {r}")
    else:
        print("  无明显有利因素")

    print("\n不利因素：")
    if reasons_negative:
        for i, r in enumerate(reasons_negative, 1):
            print(f"  {i}. ✗ {r}")
    else:
        print("  无明显不利因素")

    print("\n综合判断：")
    pos = len(reasons_positive)
    neg = len(reasons_negative)
    if survive_prob >= 0.7:
        print(f"  {name} 具备 {pos} 项有利条件，存活概率 {survive_prob*100:.1f}%，整体形势较为乐观")
    elif survive_prob >= 0.5:
        print(f"  {name} 有利与不利因素并存，存活概率 {survive_prob*100:.1f}%，结果存在较大不确定性")
    elif survive_prob >= 0.3:
        print(f"  {name} 不利因素偏多（{neg} 项），存活概率 {survive_prob*100:.1f}%，形势不容乐观")
    else:
        print(f"  {name} 具备 {neg} 项不利条件，存活概率仅 {survive_prob*100:.1f}%，处于高度危险状态")

def predict_passenger(model):
    print("\n===== 乘客存活预测 =====")
    print("请输入乘客信息：")

    name = ""
    while not name:
        name = input("乘客姓名（必填）：").strip()
        if not name:
            print("姓名不能为空，请重新输入。")
    pclass = int(input("船票等级（1/2/3）："))
    sex = input("性别（male/female）：").strip().lower()
    age = float(input("年龄："))
    sibsp = int(input("兄弟姐妹/配偶数量："))
    parch = int(input("父母/子女数量："))
    fare = float(input("票价："))
    embarked = input("登船港口（S/C/Q）：").strip().upper()
    cabin = input("舱位编号（没有请直接回车）：").strip()

    if not name:
        name = "该乘客"
    if not cabin:
        cabin = np.nan

    passenger_raw = pd.DataFrame([{
        "PassengerId": 0,
        "Name": name,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "UNKNOWN",
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked
    }])

    passenger_fe = engineer_features(passenger_raw.copy())
    passenger_X = select_features(passenger_fe)

    prob = model.predict_proba(passenger_X)[0]
    survive_prob = prob[1]
    pred = int(survive_prob >= 0.5)

    print("\n===== 预测结果 =====")
    print(f"存活概率: {survive_prob * 100:.2f}%")
    print("预测结论: {}".format("存活" if pred == 1 else "遇难"))
    analyze_reasons(name, passenger_fe.iloc[0], survive_prob)

def main():
    train, test = load_data()

    train = engineer_features(train)
    test = engineer_features(test)

    X = select_features(train)
    y = train["Survived"]
    X_test = select_features(test)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, y.index, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {X_train.shape[0]} 条")
    print(f"验证集大小: {X_val.shape[0]} 条")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val, X.columns)
    cross_validate_model(X, y)
    error_analysis(model, X_val, y_val, train, idx_val)

    print("\n===== XGBoost 交叉验证 =====")
    xgb_params = dict(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    xgb_model = XGBClassifier(**xgb_params)
    xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring="accuracy")
    print(f"每折准确率：{xgb_scores.round(4)}")
    print(f"平均准确率：{xgb_scores.mean():.4f}")
    print(f"标准差：{xgb_scores.std():.4f}")

    # 全量训练最终模型
    final_model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    final_model.fit(X, y)
    generate_submission(final_model, X_test, test)

    # 可视化
    print("\n===== 生成可视化 =====")
    feature_names = X.columns.tolist()
    plot_feature_importance(final_model, feature_names)
    plot_survival_rate(train)
    plot_radar(final_model, feature_names)
    plot_model_comparison()

    # 乘客预测
    predict_passenger(final_model)



if __name__ == "__main__":
    main()

