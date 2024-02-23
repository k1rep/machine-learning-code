import numpy as np


def divide_on_feature(X, feature_i, threshold):
    if isinstance(threshold, (int, float)):
        mask = X[:, feature_i] >= threshold
    else:
        mask = X[:, feature_i] == threshold
    return X[mask], X[~mask]


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def calculate_variance(y):
    mean = np.ones(np.shape(y)) * y.mean(axis=0)
    n_samples = np.shape(y)[0]
    variance = (1 / n_samples) * np.diag((y - mean).T.dot(y - mean))
    return variance


class SquareLoss:
    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return - (y - y_pred)


class SoftMaxLoss:
    def gradient(self, y, p):
        return y - p


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


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_total = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_total - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)


class GBDT:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 min_samples_split=3, min_impurity=1e-7, regression=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.regression = regression
        # 分类模型也使用回归树，按概率分类
        self.loss = SquareLoss() if self.regression else SoftMaxLoss()

        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(RegressionTree(min_samples_split=self.min_samples_split, min_impurity=self.min_impurity,
                                             max_depth=self.max_depth))

    def fit(self, X, y):
        self.trees[0].fit(X, y)
        y_pred = self.trees[0].predict(X)
        for i in range(1, self.n_estimators):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_pred = self.trees[0].predict(X)
        for i in range(1, self.n_estimators):
            update = self.trees[i].predict(X)
            y_pred -= np.multiply(self.learning_rate, update)
        if not self.regression:
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GBClassifier(GBDT):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 min_samples_split=2, min_impurity=1e-7):
        super(GBClassifier, self).__init__(n_estimators=n_estimators, max_depth=max_depth,
                                           learning_rate=learning_rate, min_samples_split=min_samples_split,
                                           min_impurity=min_impurity, regression=False)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GBClassifier, self).fit(X, y)


class GBDTRegressor(GBDT):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 min_samples_split=2, min_impurity=1e-7):
        super(GBDTRegressor, self).__init__(n_estimators=n_estimators, max_depth=max_depth,
                                            learning_rate=learning_rate, min_samples_split=min_samples_split,
                                            min_impurity=min_impurity, regression=True)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    data = datasets.load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    clf = GBClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("GBDT classification accuracy", classification_report(y_test, predictions))
