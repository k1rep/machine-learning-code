import numpy as np


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def divide_on_feature(X, feature_i, threshold):
    if isinstance(threshold, (int, float)):
        mask = X[:, feature_i] >= threshold
    else:
        mask = X[:, feature_i] == threshold
    return X[mask], X[~mask]


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
        self._leaf_value_calculation = None  # 计算叶子节点值的方法, 回归树为均值, 分类树为概率
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
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class XGBoostTree(DecisionTree):
    # 节点分裂
    def _split(self, y):
        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    # 计算增益
    def _gain(self, y, y_pred):
        Gradient = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
        Hessian = self.loss.hess(y, y_pred).sum()
        return 0.5 * (Gradient / Hessian)

    # 计算树分裂增益
    def _gain_by_taylor(self, y, y1, y2):
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    # 计算叶子节点最优权重
    def _approximate_update(self, y):
        y, y_pred = self._split(y)
        # Newton's Method
        gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = gradient / hessian
        return update_approximation

    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostTree, self).fit(X, y)


class CrossEntropy:
    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def gradient(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y / y_pred) + (1 - y) / (1 - y_pred)

    def hess(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return y / (y_pred ** 2) + (1 - y) / ((1 - y_pred) ** 2)


class XGBoost:
    def __init__(self, n_estimators=200, learning_rate=0.001, max_depth=2, min_samples_split=2, min_impurity=1e-7):
        self.n_estimators = n_estimators  # 树的数量
        self.learning_rate = learning_rate  # 学习率
        self.max_depth = max_depth  # 树的最大深度
        self.min_samples_split = min_samples_split  # 节点分裂最小样本数
        self.min_impurity = min_impurity  # 节点分裂最小增益

        self.loss = CrossEntropy()  # 损失函数

        self.estimators = []
        for _ in range(n_estimators):
            tree = XGBoostTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth,
                loss=self.loss
            )
            self.estimators.append(tree)

    def fit(self, X, y):
        y = to_categorical(y)
        y_pred = np.zeros(np.shape(y))
        for i in range(self.n_estimators):
            tree = self.estimators[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            update_pred = tree.predict(X)
            y_pred -= np.multiply(self.learning_rate, update_pred)

    def predict(self, X):
        y_pred = None
        for tree in self.estimators:
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred -= np.multiply(self.learning_rate, update_pred)
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = datasets.load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = XGBoost(n_estimators=500, learning_rate=0.01, max_depth=5)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)