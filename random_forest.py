import numpy as np


def divide_on_feature(X, feature_i, threshold):
    if isinstance(threshold, (int, float)):
        mask = X[:, feature_i] >= threshold
    else:
        mask = X[:, feature_i] == threshold
    return X[mask], X[~mask]


def calculate_entropy(y):
    class_labels = np.unique(y)
    entropy = 0
    for label in class_labels:
        p = len(y[y == label]) / len(y)
        entropy -= p * np.log2(p)
    return entropy


class DecisionTreeNode:
    def __init__(self, feature_i=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i  # 特征索引
        self.threshold = threshold  # 分裂阈值
        self.value = value  # 叶子节点值
        self.true_branch = true_branch  # 左子树
        self.false_branch = false_branch  # 右子树


class DecisionTree:
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float('inf'), loss=None):
        self.root = None  # 根节点
        self.min_samples_split = min_samples_split  # 节点分裂最小样本数
        self.min_impurity = min_impurity  # 节点分裂最小增益
        self.max_depth = max_depth  # 最大深度
        self._impurity_calculation = None  # 计算增益的方法, 基尼系数, 信息熵, 方差
        self._leaf_value_calculation = None  # 计算叶子节点值的方法, 回归树为均值, 分类树为最多的类别
        self.one_dim = None  # 是否是一维数据
        self.loss = loss  # 如果是Gradient Boost, 则需要传入损失函数

    def fit(self, X, y):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        largest_impurity = 0
        best_criteria = None  # 特征索引和阈值
        best_sets = None  # 左右子树样本集
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # 计算每个feature的impurity
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        impurity = self._impurity_calculation(y, y1, y2)
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # 左子树特征
                                "lefty": Xy1[:, n_features:],  # 左子树标签
                                "rightX": Xy2[:, :n_features],  # 右子树特征
                                "righty": Xy2[:, n_features:]  # 右子树标签
                            }
        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionTreeNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"],
                                    true_branch=true_branch, false_branch=false_branch)
        leaf_value = self._leaf_value_calculation(y)
        return DecisionTreeNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            print("%sT->" % indent, end="")
            self.print_tree(tree.true_branch, indent + indent)
            print("%sF->" % indent, end="")
            self.print_tree(tree.false_branch, indent + indent)


class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)
        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=float('inf'), min_samples_split=2, min_gain=0, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_features = max_features
        self.trees = [ClassificationTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth,
                                         min_impurity=self.min_gain) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        if not self.max_features:
            self.max_features = n_features
        subsets = self._get_random_subsets(X, y, self.n_estimators)
        for i in range(self.n_estimators):
            X_subset, y_subset = subsets[i]
            self.trees[i].fit(X_subset, y_subset)
            print("Tree", i, "fit complete")

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            y_preds[:, i] = tree.predict(X)
        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred

    def _get_random_subsets(self, X, y, n_subsets):
        n_samples = np.shape(X)[0]
        y = y.reshape(n_samples, 1)
        X_y = np.concatenate((X, y), axis=1)
        np.random.shuffle(X_y)
        subsets = []
        for _ in range(n_subsets):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X = X_y[idx][:, :-1]
            y = X_y[idx][:, -1]
            subsets.append([X, y])
        return subsets


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    clf = RandomForest(n_estimators=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
