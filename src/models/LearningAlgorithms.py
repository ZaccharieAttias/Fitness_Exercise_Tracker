import copy

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


class ClassificationAlgorithms:
    """
    A collection of machine learning classification algorithms with methods for
    performing feature selection and training models. This class provides implementations
    for decision trees, neural networks, support vector machines, k-nearest neighbors,
    naive bayes, and random forest classifiers, among others.
    """

    def forward_selection(self, max_features, X_train, y_train):
        """
        Perform forward selection to identify the best features for classification.

        Parameters:
        max_features (int): The maximum number of features to select.
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels.

        Returns:
        tuple: A tuple containing the selected features, ordered features, and their corresponding scores.
        """
        ordered_features = []
        ordered_scores = []
        selected_features = []
        ca = ClassificationAlgorithms()
        prev_best_perf = 0

        # Select the appropriate number of features.
        for i in range(0, max_features):
            print(f"Selecting feature {i + 1}/{max_features}")

            # Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = 0
            best_feature = None

            # For all features we can still select...
            for feature in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(feature)

                # Determine the accuracy of a decision tree learner if we were to add
                # the feature.
                model, pred_y_train, _, _, _ = ca.decision_tree(
                    X_train[temp_selected_features],
                    y_train,
                    X_train[temp_selected_features],
                )
                perf = accuracy_score(y_train, pred_y_train)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = feature

            # We select the feature with the best performance.
            if best_feature is not None:
                selected_features.append(best_feature)
                ordered_features.append(best_feature)
                ordered_scores.append(best_perf)
                prev_best_perf = best_perf

        return selected_features, ordered_features, ordered_scores

    def feedforward_neural_network(
        self,
        train_X,
        train_y,
        test_X,
        hidden_layer_sizes=(100,),
        max_iter=2000,
        activation="logistic",
        alpha=0.0001,
        learning_rate="adaptive",
        gridsearch=True,
        print_model_details=False,
    ):
        """
        Train and apply a feedforward neural network for classification.

        Parameters:
        train_X (pd.DataFrame): Training data features.
        train_y (pd.Series): Training data labels.
        test_X (pd.DataFrame): Test data features.
        hidden_layer_sizes (tuple): Sizes of hidden layers.
        max_iter (int): Maximum number of iterations.
        activation (str): Activation function for the hidden layer.
        alpha (float): L2 penalty (regularization term) parameter.
        learning_rate (str): Learning rate schedule for weight updates.
        gridsearch (bool): Whether to perform grid search for hyperparameter tuning.
        print_model_details (bool): Whether to print model details after training.

        Returns:
        tuple: Predictions and probabilities for training and test sets.
        """
        if gridsearch:
            tuned_parameters = [
                {
                    "hidden_layer_sizes": [
                        (5,),
                        (10,),
                        (25,),
                        (100,),
                        (100, 5),
                        (100, 10),
                    ],
                    "activation": [activation],
                    "learning_rate": [learning_rate],
                    "max_iter": [1000, 2000],
                    "alpha": [alpha],
                }
            ]
            nn = GridSearchCV(
                MLPClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            # Create the model
            nn = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                max_iter=max_iter,
                learning_rate=learning_rate,
                alpha=alpha,
            )

        # Fit the model
        nn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_prob_training_y = nn.predict_proba(train_X)
        pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

        return (
            nn,
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )

    def support_vector_machine_with_kernel(
        self,
        train_X,
        train_y,
        test_X,
        kernel="rbf",
        C=1,
        gamma=1e-3,
        gridsearch=True,
        print_model_details=False,
    ):
        """
        Train and apply a support vector machine (SVM) with kernel for classification.

        Parameters:
        train_X (pd.DataFrame): The training data with features.
        train_y (pd.Series): The training labels.
        test_X (pd.DataFrame): The test data for prediction.
        kernel (str): The kernel function to use ('rbf' or 'poly').
        C (float): Regularization parameter.
        gamma (float): Kernel coefficient.
        gridsearch (bool): Whether to use grid search for hyperparameter tuning.
        print_model_details (bool): Whether to print model details.

        Returns:
        tuple: Predictions and probabilities for both training and test data.
        """
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {"kernel": ["rbf", "poly"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]}
            ]
            svm = GridSearchCV(
                SVC(probability=True), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            svm = SVC(
                C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000
            )

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        pred_prob_training_y = svm.predict_proba(train_X)
        pred_prob_test_y = svm.predict_proba(test_X)
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return (
            svm,
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )

    def support_vector_machine_without_kernel(
        self,
        train_X,
        train_y,
        test_X,
        C=1,
        tol=1e-3,
        max_iter=1000,
        gridsearch=True,
        print_model_details=False,
    ):
        """
        Apply a support vector machine (SVM) classifier without a kernel.

        Parameters:
        train_X (pd.DataFrame): Training data features.
        train_y (pd.Series): Training data labels.
        test_X (pd.DataFrame): Test data features.
        C (float): Regularization parameter.
        max_iter (int): Maximum number of iterations.
        gridsearch (bool): Whether to perform grid search for hyperparameter tuning.
        print_model_details (bool): Whether to print model details after training.

        Returns:
        tuple: Predictions and probabilities for training and test sets.
        """
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {"max_iter": [1000, 2000], "tol": [1e-3, 1e-4], "C": [1, 10, 100]}
            ]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring="accuracy")
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model

        distance_training_platt = 1 / (1 + np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = (
            distance_training_platt / distance_training_platt.sum(axis=1)[:, None]
        )
        distance_test_platt = 1 / (1 + np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = (
            distance_test_platt / distance_test_platt.sum(axis=1)[:, None]
        )
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return (
            svm,
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )

    def k_nearest_neighbor(
        self,
        train_X,
        train_y,
        test_X,
        n_neighbors=5,
        gridsearch=True,
        print_model_details=False,
    ):
        """
        Apply a k-nearest neighbor classifier.

        Parameters:
        train_X (pd.DataFrame): Training data features.
        train_y (pd.Series): Training data labels.
        test_X (pd.DataFrame): Test data features.
        n_neighbors (int): Number of neighbors to use.
        weights (str): Weight function used in prediction ("uniform" or "distance").
        gridsearch (bool): Whether to perform grid search for hyperparameter tuning.
        print_model_details (bool): Whether to print model details after training.

        Returns:
        tuple: Predictions and probabilities for training and test sets.
        """
        # Create the model
        if gridsearch:
            tuned_parameters = [{"n_neighbors": [1, 2, 5, 10]}]
            knn = GridSearchCV(
                KNeighborsClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_prob_training_y = knn.predict_proba(train_X)
        pred_prob_test_y = knn.predict_proba(test_X)
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

        return (
            knn,
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )

    def decision_tree(
        self,
        train_X,
        train_y,
        test_X,
        min_samples_leaf=50,
        criterion="gini",
        print_model_details=False,
        export_tree_path="Example_graphs/Chapter7/",
        export_tree_name="tree.dot",
        gridsearch=True,
    ):
        """
        Apply a decision tree classifier.

        Parameters:
        train_X (pd.DataFrame): Training data features.
        train_y (pd.Series): Training data labels.
        test_X (pd.DataFrame): Test data features.
        criterion (str): The function to measure the quality of a split ("gini" or "entropy").
        max_depth (int or None): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        gridsearch (bool): Whether to perform grid search for hyperparameter tuning.
        print_model_details (bool): Whether to print model details after training.

        Returns:
        tuple: Predictions and probabilities for training and test sets.
        """
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "criterion": ["gini", "entropy"],
                }
            ]
            dtree = GridSearchCV(
                DecisionTreeClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            dtree = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf, criterion=criterion
            )

        # Fit the model

        dtree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(
            pred_prob_training_y, columns=dtree.classes_
        )
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(dtree.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            print("Feature importance decision tree:")
            for i in range(0, len(dtree.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(dtree.feature_importances_[ordered_indices[i]])
            tree.export_graphviz(
                dtree,
                out_file=export_tree_path + export_tree_name,
                feature_names=train_X.columns,
                class_names=dtree.classes_,
            )

        return (
            dtree,
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )

    def naive_bayes(self, train_X, train_y, test_X):
        """
        Apply a naive bayes classifier.

        Parameters:
        train_X (pd.DataFrame): Training data features.
        train_y (pd.Series): Training data labels.
        test_X (pd.DataFrame): Test data features.
        var_smoothing (float): Portion of the largest variance of all features added to variances for stability.
        gridsearch (bool): Whether to perform grid search for hyperparameter tuning.
        print_model_details (bool): Whether to print model details after training.

        Returns:
        tuple: Predictions and probabilities for training and test sets.
        """
        # Create the model
        nb = GaussianNB()

        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(test_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

        return (
            nb,
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )

    def random_forest(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=10,
        min_samples_leaf=5,
        criterion="gini",
        print_model_details=False,
        gridsearch=True,
    ):
        """
        Apply a random forest classifier for classification.

        Parameters:
        train_X (pd.DataFrame): Training data features.
        train_y (pd.Series): Training data labels.
        test_X (pd.DataFrame): Test data features.
        n_estimators (int): The number of trees in the forest.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        criterion (str): The function to measure the quality of a split (either "gini" or "entropy").
        print_model_details (bool): Whether to print model details after training.
        gridsearch (bool): Whether to perform grid search for hyperparameter tuning.

        Returns:
        tuple: Predictions and probabilities for training and test sets.
        """
        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 50, 100],
                    "criterion": ["gini", "entropy"],
                }
            ]
            rf = GridSearchCV(
                RandomForestClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
            )

        # Fit the model

        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True
                )
            ]
            print("Feature importance random forest:")
            for i in range(0, len(rf.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(rf.feature_importances_[ordered_indices[i]])

        return (
            rf,
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
        )
