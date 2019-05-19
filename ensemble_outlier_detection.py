# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV

method_name = 'pls'  # 'pls' or 'svr'
number_of_submodels = 100  # サブモデルの数
max_iteration_number = 30  # 繰り返し回数

fold_number = 2  # N-fold CV の N
max_number_of_principal_components = 30  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

dataset = pd.read_csv('numerical_simulation_data.csv', index_col=0)
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

# 初期化
outlier_sample_flags = ~(y == y)
previous_outlier_sample_flags = ~(y == y)
for iteration_number in range(max_iteration_number):
    print(iteration_number + 1, '/', max_iteration_number)  # 進捗状況の表示
    normal_x = x[~outlier_sample_flags]
    normal_y = y[~outlier_sample_flags]
        
    estimated_y_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとの y の推定結果を追加
    for submodel_number in range(number_of_submodels):
#        print(submodel_number + 1, '/', number_of_submodels)  # 進捗状況の表示
        # 説明変数の選択
        # 0 から (サンプル数) までの間に一様に分布する乱数をサンプルの数だけ生成して、その floor の値の番号のサンプルを選択
        selected_sample_numbers = np.floor(np.random.rand(normal_x.shape[0]) * normal_x.shape[0]).astype(int)
        selected_x = normal_x.iloc[selected_sample_numbers, :]
        selected_y = normal_y.iloc[selected_sample_numbers]
        unique_number, unique_index = np.unique(selected_sample_numbers, return_index=True)
        # オートスケーリング
        selected_autoscaled_x = (selected_x - selected_x.mean()) / selected_x.std()
        autoscaled_x = (x - selected_x.mean()) / selected_x.std()
        selected_autoscaled_y = (selected_y - selected_y.mean()) / selected_y.std()
        
        if method_name == 'pls':
            # CV による成分数の最適化
            components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
            r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
            for component in range(1, min(np.linalg.matrix_rank(selected_autoscaled_x),
                                          max_number_of_principal_components) + 1):
                # PLS
                submodel_in_cv = PLSRegression(n_components=component)  # PLS モデルの宣言
                estimated_y_in_cv = pd.DataFrame(cross_val_predict(submodel_in_cv, selected_autoscaled_x, selected_autoscaled_y,
                                                                   cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
                estimated_y_in_cv = estimated_y_in_cv * selected_y.std() + selected_y.mean()  # スケールをもとに戻す
                r2_in_cv = metrics.r2_score(selected_y, estimated_y_in_cv)  # r2 を計算
                r2_in_cv_all.append(r2_in_cv)  # r2 を追加
                components.append(component)  # 成分数を追加
            optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]
            # PLS
            submodel = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
        elif method_name == 'svr':
            # ハイパーパラメータの最適化
            # グラム行列の分散を最大化することによる γ の最適化
            variance_of_gram_matrix = list()
            for svr_gamma in svr_gammas:
                gram_matrix = np.exp(
                    -svr_gamma * cdist(selected_autoscaled_x, selected_autoscaled_x, metric='seuclidean'))
                variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
            optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
            # CV による ε の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                                       cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, selected_autoscaled_y)
            optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
            # CV による C の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                                       {'C': svr_cs}, cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, selected_autoscaled_y)
            optimal_svr_c = model_in_cv.best_params_['C']
            # CV による γ の最適化
            model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                                       {'gamma': svr_gammas}, cv=fold_number, iid=False)
            model_in_cv.fit(selected_autoscaled_x, selected_autoscaled_y)
            optimal_svr_gamma = model_in_cv.best_params_['gamma']
            # SVR
            submodel = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon,
                               gamma=optimal_svr_gamma)  # モデルの宣言
        submodel.fit(selected_autoscaled_x, selected_autoscaled_y)  # モデルの構築
        estimated_y = np.ndarray.flatten(submodel.predict(autoscaled_x))  # 推定

        estimated_y = pd.DataFrame(estimated_y, columns=['{0}'.format(submodel_number)])  # テストデータの y の値を推定し、Pandas の DataFrame 型に変換
        estimated_y = estimated_y * selected_y.std() + selected_y.mean()  # スケールをもとに戻します
        estimated_y_all = pd.concat([estimated_y_all, estimated_y], axis=1)
    estimated_y_all.index = y.index
    
    # 外れサンプルの判定
    estimated_y_all_normal = estimated_y_all[~outlier_sample_flags]
    estimated_y_median_normal = estimated_y_all_normal.median(axis=1)
    estimated_y_mad_normal = np.median(abs(estimated_y_all_normal - np.median(estimated_y_median_normal)))
    y_error = abs(y - estimated_y_all.median(axis=1))
    outlier_sample_flags = y_error > 3 * 1.4826 * estimated_y_mad_normal
    print('外れサンプル検出結果が一致した数 :', sum(outlier_sample_flags == previous_outlier_sample_flags))
    if sum(outlier_sample_flags == previous_outlier_sample_flags) == x.shape[0]:
        print('計算終了')
        break
    previous_outlier_sample_flags = outlier_sample_flags.copy()
    
outlier_sample_flags = pd.DataFrame(outlier_sample_flags)
outlier_sample_flags.columns = ['TRUE if outlier samples']
outlier_sample_flags.to_csv('outlier_sample_detection_results.csv')
