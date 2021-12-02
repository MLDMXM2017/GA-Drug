# -*- coding: UTF-8 -*-
"""
基本GA，个体中的基因用于特征选择
v15:3个并行GA
"""
import numpy as np
import pandas as pd
import readData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from xgboost import XGBClassifier
import heapq
import copy
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

###########忽略警告
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################
# Parameter Setting
########################################################################################################################
generation_size = 100
pop_size = 50
gene_size = 1956
pc = 0.5    # possibility of crossover in GA
pm = 0.1   # possibility of mutation in GA
split_num = 3   # the number of large scale of instance
runtimes = 5
random_states = np.random.randint(0,100,size=20)

class_0_p = 0.622
class_1_p = 0.1849
class_2_p = 0.1931


########################################################################################################################
# Function
########################################################################################################################

def evaluate(train_X, train_y, test_X, test_y):
    # base_estimator = KNeighborsClassifier()
    # base_estimator = LogisticRegression()
    # base_estimator = RandomForestClassifier(n_estimators=50, random_state=0)
    base_estimator = ExtraTreesClassifier(n_estimators=100,random_state=0)
    # base_estimator = XGBClassifier(learning_rate=0.1, n_estimators=250, random_state=0)
    # base_estimator = tree.DecisionTreeClassifier()
    # base_estimator = GradientBoostingClassifier()
    # base_estimator = AdaBoostClassifier()
    # base_estimator = QuadraticDiscriminantAnalysis()
    # base_estimator = GaussianNB()
    # base_estimator = SVC()
    # base_estimator = LinearDiscriminantAnalysis()
    # base_estimator = gcForest(shape_1X=[1,train_X.shape[1]], window=3)
    base_estimator.fit(train_X, train_y)
    predict_y = base_estimator.predict(test_X)
    result = f1_score(test_y, predict_y, average='macro')
    return result, predict_y


def getDataByFS(trainX, testX, individual):
    '''
    输出通过个体特征选择后的数据
    :param trainX:
    :param testX:
    :param individual:
    :return:
    '''
    fs = []
    for i,item in enumerate(individual):
        if item == 1:
            fs.append(i)
            # fs.append(i+gene_size)  # 对应两个978的特征，同时选择或者同时不选择
    return trainX[:,fs], testX[:,fs]


def vote(ylist, values):
    '''
    个体中的3个cell进行投票
    :param ylist: 需要进行投票的3个cell的predict_y
    :param values: 3个cell的f1-score
    :return: 投票出的ylabel
    '''
    dict = {0:0, 1:0, 2:0}  # ylabel所对应的票数字典
    for y in ylist:
        dict[y] = dict[y] + 1
    maxvote = max(dict, key=dict.get)
    if dict[maxvote] == 1:
        # 平票PK
        maximum = max(values)
        if values[0] == maximum:
            return 0
        elif values[1] == maximum:
            return 1
        else:
            return 2
    else:
        return maxvote

def vote_ova(ylist, values):
    dict = {'P':0, 'N':0}   # ova数据的输出分布
    for y in ylist:
        dict[y] = dict[y] + 1
    if dict['P'] == 1:
        return ylist.tolist().index('P'), False
    elif dict['P'] == 0 :
        # 3个类的OVA分类器都不要，分配给f1score最小的类
        # return values.index(min(values))
        # 3个类的OVA分类器都不要，按样本数量概率轮盘分配
        random_number = np.random.randint(0,1)
        if random_number <= class_0_p:
            return 0, True
        elif random_number <= class_1_p:
            return 1, True
        else:
            return 2, True
    elif dict['P'] == 2:
        # 有2个OVA分类器都要，比较f1score，分配给f1score最大的类
        first_index = ylist.tolist().index('P')
        second_index = ylist.tolist().index('P', first_index+1)
        if values[first_index] > values[second_index]:
            return first_index, True
        else:
            return second_index, True
    else:
        # 有3个OVA分类器都要，分配给f1score最大的类
        return values.index(max(values)), True

def vote_v3(ylist, code_proba_dict):
    '''
    更改平票策略，当平票时根据编码概率进行分配
    :param ylist:
    :param code_proba_dict:
    :return:
    '''
    dict = {'P': 0, 'N': 0}
    for y in ylist:
        dict[y] += 1
    if dict['P'] == 1:
        return ylist.index('P'), False
    else:
        # 平票
        ylist_code = ylist[0] + ylist[1] + ylist[2]
        proba_list = code_proba_dict[ylist_code]
        return proba_list.index(max(proba_list)), True

def vote_by_distance(predictY, X, individual):
    '''
    根据样本与类中心的距离进行分类器的选择
    :param predictY: 所有样本3个分类器的输出
    :param X: 所有样本
    :param individual: 特征选择矩阵
    :return:
    '''
    select_cell_list = []
    for i in range(len(X)):
        select_cell_list.append(get_clustering_label(X[i],individual))

    selected_y = []
    for i in range(len(select_cell_list)):
        dict = {0: 0, 1: 0, 2: 0}  # ylabel所对应的票数字典
        for y in predictY[:,i]:
            dict[y] = dict[y] + np.exp(-select_cell_list[i])
        maxvote = max(dict, key=dict.get)
        # # 距离最近的分类器结果为P则直接取这个分类器结果
        # if predictY[select_cell_list[i][0],i] == 'P':
        #     selected_y.append(select_cell_list[i][0])
        # # 若距离最近的分类器结果为N，则考察距离第二近的分类器结果
        # elif predictY[select_cell_list[i][1],i] == 'P':
        #     selected_y.append(select_cell_list[i][1])
        # # 若距离第二近的分类结果为N，则考察距离第三近的分类器结果
        # elif predictY[select_cell_list[i][2],i] == 'P':
        #     selected_y.append(select_cell_list[i][2])
        # # 若三个分类器分类结果都为N，则分配给f1score最小的类
        # else:
        #     values.index(min(values))
        # selected_y.append(predictY[select_cell_list[i],i])
        selected_y.append(maxvote)

    return selected_y

def vote_by_classifier(predictY, train_predict_y, train_y):
    '''
    对分类器结果进行训练
    :param predictY:
    :param train_X:
    :param train_y:
    :return:
    '''
    base_estimator = RandomForestClassifier(n_estimators=50, random_state=0)
    base_estimator.fit(np.array(train_predict_y).T, train_y)
    voted_y = base_estimator.predict(np.array(predictY).T)
    return voted_y

def get_diversity(predict_y_list):
    '''
    返回3个OVA cell之间的差异值，差异为每两个cell之间P N的总和
    :param predict_y_list:
    :return:
    '''
    diversity = 0
    for i in range(3):
        cell_index = [0, 1, 2]  # 三个cell的索引
        cell_index.remove(i)  #遍历cell对
        for y1,y2 in zip(predict_y_list[cell_index[0]], predict_y_list[cell_index[1]]):
            if y1 == 'P' and y2 == 'N':
                diversity = diversity + 1
            elif y1 == 'N' and y2 == 'P':
                diversity = diversity + 1
    return diversity

def get_CE_diversity(predict_y_list, test_y):
    '''
    返回所有样本中3个OVA cell之间的Correct Error diversity，correct代表该分类器分类正确，error代表该分类器分类错误
    :param predict_y_list:
    :param test_y:
    :return:
    '''
    diversity = 0
    predict_y_list = np.array(predict_y_list)
    for i in range(len(test_y)):
        if sum(predict_y_list[:,i] == 'P') == 3:
            diversity = diversity + 1
        elif sum(predict_y_list[:,i] == 'P') == 2:
            if predict_y_list[int(test_y[i]),i] != 'N':
                diversity = diversity + 1
    return diversity

def get_three_label_diversity(predict_y_list):
    '''
    返回3个cell（非OVA）对之间的diversity，只要结果不同就算一个diversity样本
    :param predict_y_list:
    :return:
    '''
    diversity = 0
    for i in range(split_num):
        cell_index = [0, 1, 2]  # 三个cell的索引
        cell_index.remove(i)  # 遍历cell对
        for y1, y2 in zip(predict_y_list[cell_index[0]], predict_y_list[cell_index[1]]):
            if y1 != y2:
                diversity = diversity + 1
    return diversity

def get_recall_variance(test_y, vote_y):
    '''
    返回个体3个cell投票完后的y的3类的recall方差
    :param test_y:
    :param vote_y:
    :return:
    '''
    classification_report = metrics.classification_report(test_y, vote_y, output_dict=True)
    var = np.var([classification_report['0.0']['recall'],
                  classification_report['1.0']['recall'],
                  classification_report['2.0']['recall']])
    std = np.std([classification_report['0.0']['recall'],
                  classification_report['1.0']['recall'],
                  classification_report['2.0']['recall']],ddof=1)
    return (2*(classification_report['1.0']['recall']*classification_report['2.0']['recall'])) / (classification_report['1.0']['recall'] + classification_report['2.0']['recall'])

def get_fitness(train_X, train_y, test_X, test_y, individual, return_cell_values=False, ova_type=0):
    '''
    一个个体的评价函数
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :param individual:
    :param return_cell_values: 是否输出个体中3个cell的分数
    :return:
    '''
    selectedTrainX, selectedTestX = getDataByFS(
        train_X[ova_type],
        test_X,
        individual)
    ova_train_y = get_OVA_data(train_y[ova_type], ova_type)
    ova_test_y = get_OVA_data(test_y, ova_type)
    value, predict_y = evaluate(selectedTrainX, ova_train_y, selectedTestX, ova_test_y)

    if return_cell_values == True:
        return value, predict_y

    return value, predict_y


def output_selected_data(selectedTrainX, trainy, selectedValidateX, validatey, selectedTestX, testy):
    f = open('./selectedData.csv', 'w')
    csv_writer = csv.writer(f,lineterminator='\n')
    for i in range(len(selectedTrainX)):
        csv_writer.writerow(np.hstack((selectedTrainX[i],trainy[i])))
    for i in range(len(selectedValidateX)):
        csv_writer.writerow(np.hstack((selectedTrainX[i],validatey[i])))
    for i in range(len(selectedTestX)):
        csv_writer.writerow(np.hstack((selectedTrainX[i],testy[i])))
    f.close()

def output_results(validate_results, test_results, g_test_results):
    '''
    输出runtimes次GA generation_size代迭代 所有的validate values和test values
    :param validate_results:
    :param test_results:
    :return:
    '''
    f = open('./results.csv', 'a')
    csv_writer = csv.writer(f, lineterminator='\n')
    for i in range(runtimes):
        csv_writer.writerow(np.hstack((str(i)+" validate_results", validate_results[i])))
        csv_writer.writerow(np.hstack((str(i)+" test_results", test_results[i])))
        csv_writer.writerow(np.hstack((str(i) + " g_test_results", g_test_results[i])))
    f.close()

def output_error_correct(runtime, generation_index, type, ec_list):
    if type == "validate":
        f = open('./validate_error_correct', 'a')
    elif type == "test":
        f = open('./test_error_correct', 'a')
    else:
        f = open('./g_test_error_correct', 'a')
    csv_writer = csv.writer(f, lineterminator='\n')
    for ec in ec_list:
        csv_writer.writerow(np.hstack((str(runtime) + " " + str(generation_index),
                                       ec)))
    f.close()

def output_cell_values(runtime, generation_index, type, test_y, voted_y, f1score, dimension_result):
    '''
    输出runtimes次GA generation_size代迭代，所有最优个体split_num个cell的分数
    :param runtime:
    :param generation_index:
    :param values:
    :param type: 需要记录的cell分数的类型，validate or test
    :return:
    '''
    if type == 'validate':
        f = open('./cell_validate_values.csv', 'a')
    elif type == 'test':
        f = open('./cell_test_values.csv', 'a')
    else:
        f = open('./cell_g_test_values.csv', 'a')
    csv_writer = csv.writer(f, lineterminator='\n')

    classification_report = metrics.classification_report(test_y, voted_y, output_dict=True)
    accuracy = metrics.accuracy_score(test_y, voted_y)

    csv_writer.writerow(np.hstack((str(runtime) + "th " + str(generation_index),
                                   classification_report['0.0']['recall'],
                                   classification_report['1.0']['recall'],
                                   classification_report['2.0']['recall'],
                                   classification_report['0.0']['precision'],
                                   classification_report['1.0']['precision'],
                                   classification_report['2.0']['precision'],
                                   accuracy,
                                   f1score,
                                   dimension_result[0],
                                   dimension_result[1],
                                   dimension_result[2])))
    f.close()

def output_individual(runtime, generation_index, individual):
    '''
    输出runtimes次GA generation_size代迭代，所有最优个体split_num个cell的特征选择向量
    :param runtime:
    :param generation_index:
    :param individual:
    :return:
    '''
    f = open('./cell_code.csv', 'a')
    csv_writer = csv.writer(f, lineterminator='\n')
    for i in range(split_num):
        csv_writer.writerow(np.hstack((str(runtime) + "th " + str(generation_index) + "generation cell" + str(i),
                             individual[i*gene_size:(i+1)*gene_size])))

def large_scale_data_split(data):
    '''
    将大类NA样本分割split_num份后与小类样本合并生成split_num份数据
    :param data: 待分割数据
    :param split_num: 分割份数
    :return:
    '''
    new_train_data = []
    data = pd.DataFrame(data)

    large_scale_data = data[data.iloc[:,-1] == 0].values
    # 重新规划0类比例
    splited_data = []
    for i in range(split_num):
        np.random.shuffle(large_scale_data)
        splited_data.append(large_scale_data)
        # if i == 0:
        #     splited_data.append(large_scale_data)
        # else:
        #     splited_data.append(large_scale_data[:int(len(large_scale_data)*0.4)])
    splited_data = np.array(splited_data)
    for i in range(split_num):
        new_train_data.append(np.vstack((data[data.iloc[:,-1] == 1].values,
                                      data[data.iloc[:,-1] == 2].values,
                                      splited_data[i])))
        np.random.shuffle(new_train_data[i])

    new_train_X = []
    new_train_y = []
    for i in range(split_num):
        new_train_X.append(new_train_data[i][:,:-1])
        new_train_y.append(new_train_data[i][:,-1])
    return new_train_X, new_train_y

def get_OVA_data(train_y, keep_class_label):
    '''
    将输入数据转换为OVA数据，保留的正类为P，负类为N
    :param train_y: 需要转换的数据label列
    :param keep_class_label: 需要保留的类标签
    :return:
    '''
    ova_data_y = []
    for i in range(len(train_y)):
        if train_y[i] == keep_class_label:
            ova_data_y.append('P')
        else:
            ova_data_y.append('N')
    return np.array(ova_data_y)

def get_clustering_label(instance,individual):
    '''
    根据样本获取样本在当前特征空间下距离哪个数据group更近，返回index
    :param instance: 样本
    :param individual: 特征选择向量
    :return:
    '''
    distance = []
    for i in range(split_num):
        temp_individual = [bool(x) for x in individual]
        distance.append(np.linalg.norm(np.array(data_group_center[i][temp_individual[i*gene_size:(i+1)*gene_size]]) -
                                       np.array(instance[temp_individual[i*gene_size:(i+1)*gene_size]])))
    min_distance = min(distance)
    index = distance.index(min_distance)
    # return sorted(enumerate(distance), key=lambda x:x[1])
    return index

def get_feature_selection_pool(train_X,train_y):
    data = pd.DataFrame(np.hstack((train_X,train_y.reshape(len(train_y),1))))

    fs_importance = []
    target_data0 = data[data.iloc[:,-1] == 0].values
    target_data1 = data[data.iloc[:, -1] == 1].values
    target_data2 = data[data.iloc[:, -1] == 2].values

    estimator = ExtraTreesClassifier(n_estimators=100, random_state=0)
    # estimator = XGBClassifier(learning_rate=0.1, n_estimators=200, random_state=0)
    estimator.fit(np.vstack((target_data0, target_data1, target_data2))[:, :-1],
                  get_OVA_data(np.vstack((target_data0,target_data1,target_data2))[:,-1], 0.0))
    fs_importance.append(estimator.feature_importances_)
    estimator.fit(np.vstack((target_data0, target_data1, target_data2))[:, :-1],
                  get_OVA_data(np.vstack((target_data0, target_data1, target_data2))[:, -1], 1.0))
    fs_importance.append(estimator.feature_importances_)
    estimator.fit(np.vstack((target_data0, target_data1, target_data2))[:, :-1],
                  get_OVA_data(np.vstack((target_data0, target_data1, target_data2))[:, -1], 2.0))
    fs_importance.append(estimator.feature_importances_)

    fs_pool = []
    for i in range(len(fs_importance)):
        temp_fs_pool = np.zeros(gene_size).tolist()
        fs_importance[i] = fs_importance[i].tolist()
        for j in range(1000):
            temp_index = fs_importance[i].index(max(fs_importance[i]))
            temp_fs_pool[temp_index] = 1
            fs_importance[i][temp_index] = 0
        fs_pool.append(temp_fs_pool)

    return fs_pool

def get_code_proba(predict_code, validate_y):
    # 代表每种编码属于3类的概率
    # OVA版本全部code
    dict = {
        'PPP': [0,0,0],
        'PPN': [0,0,0],
        'PNP': [0,0,0],
        'NPP': [0,0,0],
        'NNN': [0,0,0]
    }
    # 三分类版本平票code
    # dict = {
    #     '012': [0, 0, 0],
    #     '021': [0, 0, 0],
    #     '102': [0, 0, 0],
    #     '120': [0, 0, 0],
    #     '201': [0, 0, 0],
    #     '210': [0, 0, 0]
    # }

    predict_code = np.array(predict_code).T.tolist()
    # predict_code = [str(int(x[0])) + str(int(x[1])) + str(int(x[2])) for x in predict_code]
    predict_code = [str(x[0]) + str(x[1]) + str(x[2]) for x in predict_code]
    for i in range(len(predict_code)):
        if predict_code[i] in dict.keys():
            dict[predict_code[i]][int(validate_y[i])] += 1
    for code in list(dict.keys()):
        temp_sum = dict[code][0] + dict[code][1] + dict[code][2]
        if temp_sum != 0:
            dict[code] = [x/temp_sum for x in dict[code]]
    return dict



########################################################################################################################
# GA
########################################################################################################################
'load data'
# data = data_loader.load_data2()
# data = readData.get_data()
data = readData.get_test_data()
genalization_test_data = readData.get_test_data()

X, y = data[:,:-1], data[:,-1]
# 归一化
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

g_test_X, g_test_y = genalization_test_data[:,:-1], genalization_test_data[:,-1]
g_test_X = min_max_scaler.transform(g_test_X)

skf = StratifiedKFold(n_splits=5,shuffle=True)

'repeat GA on multiple segmentation data'
validate_results = []
test_results = []
g_test_results = []
runtime = 0
for train_index, test_index in skf.split(X,y):

    'k fold'
    train_X, test_X, train_y, test_y = X[train_index], X[test_index], y[train_index], y[test_index]
    train_X, validate_X, train_y, validate_y = train_test_split(train_X, train_y,
                                                                test_size=0.25,
                                                                random_state=random_states[runtime],
                                                                stratify=train_y)

    # 将NA类分成split_num份后和小类样本拼接
    train_X, train_y = large_scale_data_split(np.hstack((train_X,train_y.reshape(len(train_y),1))))

    'generate population and individual for GA'
    # 针对各个类别生成特定的特征选择向量后做随机改变
    # fs_pool = get_feature_selection_pool(np.vstack((train_X[0],train_X[1],train_X[2])),np.hstack((train_y[0],train_y[1],train_y[2])))
    # population = []
    # for i in range(split_num):
    #     temp_population = []
    #     for j in range(pop_size):
    #         temp_individual = copy.deepcopy(fs_pool[i])
    #         temp_index = np.random.randint(0,gene_size,size=round(gene_size*pm))
    #         for index in temp_index:
    #             temp_individual[index] = np.random.randint(0,2)
    #         temp_population.append(temp_individual)
    #     population.append(temp_population)
    # population = np.array(population)
    population = []
    for i in range(split_num):
        temp_population = []
        for j in range(pop_size):
            individual = np.random.randint(0, 2, size=gene_size)
            temp_population.append(individual)
        population.append(temp_population)
    population = np.array(population)

    # check if individual is all 0
    # for individual in population:
    #     while (individual == 0).all():
    #         individual = np.random.randint(0,2,gene_size * split_num)
    #         if (individual == 1).any():
    #             break

    'evolution'
    test_values = []
    g_test_values = []
    validate_values = []
    validate_average_values = []
    imp_feature_times = [0] * 1956
    for generation_index in range(generation_size):

        # crossover
        # 分成3个GA分别进行crossover
        children = []
        for i in range(split_num):
            parent_index = np.arange(pop_size)
            np.random.shuffle(parent_index)
            parent_left,parent_right = parent_index[0:int(len(parent_index)/2)],parent_index[int(len(parent_index)/2):]
            temp_children = []
            for (left,right) in zip(parent_left,parent_right):
                child_left,child_right = copy.deepcopy(population[i][left]),copy.deepcopy(population[i][right])
                index = np.random.randint(0,gene_size)
                temp = child_left[0:index]
                child_left[0:index] = child_right[0:index]
                child_right[0:index] = temp
                # for i in range(split_num):
                #     temp = child_left[i*gene_size:i*gene_size+index]
                #     child_left[i*gene_size:i*gene_size+index] = child_right[i*gene_size:i*gene_size+index]
                #     child_right[i*gene_size:i*gene_size+index] = temp
                temp_children.append(child_left)
                temp_children.append(child_right)
            children.append(temp_children)

        # mutation
        # 分成3个GA分别进行mutation
        for i in range(split_num):
            for j in range(pop_size):
                gene_index = np.arange(gene_size)
                np.random.shuffle(gene_index)
                gene_index = gene_index[0:int(pm*gene_size)]
                for k in gene_index:
                    if children[i][j][k] == 0:
                        children[i][j][k] = 1
                    else:
                        children[i][j][k] = 0

        # evaluation
        if generation_index == 0:
            parentResults = []
            for i in range(split_num):
                parentResult = []
                for j in range(pop_size):
                    fitness_value, predict_y = get_fitness(train_X, train_y, validate_X, validate_y, population[i][j], ova_type=i)
                    parentResult.append(
                        fitness_value
                    )
                parentResults.append(parentResult)


        childrenResults = []
        for i in range(split_num):
            childrenResult = []
            for j in range(pop_size):
                fitness_value, predict_y = get_fitness(train_X, train_y, validate_X, validate_y, children[i][j], ova_type=i)
                childrenResult.append(
                    fitness_value
                )
            childrenResults.append(childrenResult)

        # elite selection
        # 分3个维度进行精英选择
        new_population = []
        new_population_result = []
        for i in range(split_num):
            parentResult = parentResults[i]
            childrenResult = childrenResults[i]
            result = parentResult + childrenResult
            max_num_index_list = map(result.index, heapq.nlargest(pop_size, result))

            temp_new_population = []
            temp_new_population_result = []
            updated_num = 0
            for j in max_num_index_list:
                if j >= pop_size:
                    temp_new_population.append(children[i][j - pop_size])
                    temp_new_population_result.append(childrenResults[i][j - pop_size])
                    updated_num = updated_num + 1
                else:
                    temp_new_population.append(population[i][j])
                    temp_new_population_result.append(parentResults[i][j])
            print(str(i+1) + " dimension updated into population:" + str(updated_num))
            new_population.append(temp_new_population)
            new_population_result.append(temp_new_population_result)

        # 选择每个维度最好的个体一起投票
        # dimension_index = []
        # for i in range(split_num):
        #     dimension_index.append(new_population_result[i].index(max(new_population_result[i])))

        # validate
        best_dimension_result = []
        best_dimension_predictY = []
        # 3个通道都保留5个结果
        for i in range(split_num):
            temp_result, temp_predict_y = get_fitness(train_X,train_y,
                                                      validate_X, validate_y,
                                                      new_population[i][0],
                                                      ova_type=i)
            best_dimension_result.append(temp_result)
            best_dimension_predictY.append(temp_predict_y)
        print("3 dimension ova f1score : validate")
        print(best_dimension_result)

        # 求编码概率字典
        code_proba_dict = get_code_proba(best_dimension_predictY, validate_y)
        print("编码概率字典：")
        print(code_proba_dict)

        # 将每个维度最好的个体挑选出来后，将结果进行投票
        voted_y = []
        tieNumber = 0
        tieNumber_original = 0
        correctTieNumber = 0
        correctTieNumber_original = 0
        error_correct_list = []
        for i in range(len(validate_y)):
            temp_voted_y_original, isTie_original = vote_ova(np.array(best_dimension_predictY)[:,i],best_dimension_result)
            temp_voted_y, isTie = vote_v3(np.array(best_dimension_predictY)[:,i].tolist(), code_proba_dict)
            # 纠错统计
            temp_error_correct = []
            for idx, y in enumerate(np.array(best_dimension_predictY)[:,i]):
                if y == 'P':
                    temp_error_correct.append(1 if idx == validate_y[i] else 0)
                else:
                    temp_error_correct.append(0 if idx == validate_y[i] else 1)
            temp_error_correct.append(1 if temp_voted_y == validate_y[i] else 0)
            temp_error_correct.append(1 if temp_voted_y_original == validate_y[i] else 0)
            error_correct_list.append(temp_error_correct)
            voted_y.append(temp_voted_y)
            if isTie:
                tieNumber += 1
                if temp_voted_y == validate_y[i]:
                    correctTieNumber += 1
            if isTie_original:
                tieNumber_original += 1
                if temp_voted_y_original == validate_y[i]:
                    correctTieNumber_original += 1
        print("(之前的方法)平票百分比：%s" % (tieNumber_original / len(validate_y)))
        print("(之前的方法)平票正确百分比：%s" % (correctTieNumber_original / tieNumber_original))
        print("平票百分比：%s" % (tieNumber/len(validate_y)))
        print("平票正确百分比：%s" % (correctTieNumber / tieNumber))
        validate_f1score = f1_score(validate_y, voted_y, average='macro')

        validate_values.append(validate_f1score)
        output_cell_values(runtime,
                           generation_index,
                           'validate',
                           validate_y,
                           voted_y,
                           validate_f1score,
                           best_dimension_result)
        output_error_correct(runtime,
                             generation_index,
                             'validate',
                             error_correct_list)
        best_dimension_result_validate = copy.deepcopy(best_dimension_result)

        # test
        best_dimension_result = []
        best_dimension_predictY = []
        for i in range(split_num):
            temp_result, temp_predict_y = get_fitness(train_X, train_y,
                                                      test_X, test_y,
                                                      new_population[i][0],
                                                      ova_type=i)
            best_dimension_result.append(temp_result)
            best_dimension_predictY.append(temp_predict_y)
        print("3 dimension ova f1score : test")
        print(best_dimension_result)
        # 将每个维度最好的个体挑选出来后，将结果进行投票
        voted_y = []
        tieNumber = 0
        tieNumber_original = 0
        correctTieNumber = 0
        correctTieNumber_original = 0
        error_correct_list = []
        for i in range(len(test_y)):
            temp_voted_y_original, isTie_original = vote_ova(np.array(best_dimension_predictY)[:,i],best_dimension_result)
            temp_voted_y, isTie = vote_v3(np.array(best_dimension_predictY)[:, i].tolist(), code_proba_dict)
            # 纠错统计
            temp_error_correct = []
            for idx, y in enumerate(np.array(best_dimension_predictY)[:,i]):
                if y == 'P':
                    temp_error_correct.append(1 if idx == test_y[i] else 0)
                else:
                    temp_error_correct.append(0 if idx == test_y[i] else 1)
            temp_error_correct.append(1 if temp_voted_y == test_y[i] else 0)
            temp_error_correct.append(1 if temp_voted_y_original == test_y[i] else 0)
            error_correct_list.append(temp_error_correct)
            voted_y.append(temp_voted_y)
            if isTie:
                tieNumber += 1
                if temp_voted_y == test_y[i]:
                    correctTieNumber += 1
            if isTie_original:
                tieNumber_original += 1
                if temp_voted_y_original == test_y[i]:
                    correctTieNumber_original += 1
        print("(之前的方法)平票百分比：%s" % (tieNumber_original / len(test_y)))
        print("(之前的方法)平票正确百分比：%s" % (correctTieNumber_original / tieNumber_original))
        print("平票百分比：%s" % (tieNumber / len(validate_y)))
        print("平票正确百分比：%s" % (correctTieNumber / tieNumber))
        test_f1score = f1_score(test_y, voted_y, average='macro')

        test_values.append(test_f1score)
        output_cell_values(runtime,
                           generation_index,
                           'test',
                           test_y,
                           voted_y,
                           test_f1score,
                           best_dimension_result)
        output_error_correct(runtime,
                             generation_index,
                             'test',
                             error_correct_list)
        # output_individual(runtime, generation_index, new_population[0])

        # 泛化效果测试数据
        best_dimension_result = []
        best_dimension_predictY = []
        for i in range(split_num):
            temp_result, temp_predict_y = get_fitness(train_X, train_y,
                                                      g_test_X, g_test_y,
                                                      new_population[i][0],
                                                      ova_type=i)
            best_dimension_result.append(temp_result)
            best_dimension_predictY.append(temp_predict_y)
        print("3 dimension ova f1score : g_test")
        print(best_dimension_result)
        # 将每个维度最好的个体挑选出来后，将结果进行投票
        voted_y = []
        tieNumber = 0
        tieNumber_original = 0
        correctTieNumber = 0
        correctTieNumber_original = 0
        error_correct_list = []
        for i in range(len(g_test_y)):
            temp_voted_y_original, isTie_original = vote_ova(np.array(best_dimension_predictY)[:, i],
                                                             best_dimension_result)
            temp_voted_y, isTie = vote_v3(np.array(best_dimension_predictY)[:, i].tolist(), code_proba_dict)
            # 纠错统计
            temp_error_correct = []
            for idx, y in enumerate(np.array(best_dimension_predictY)[:, i]):
                if y == 'P':
                    temp_error_correct.append(1 if idx == g_test_y[i] else 0)
                else:
                    temp_error_correct.append(0 if idx == g_test_y[i] else 1)
            temp_error_correct.append(1 if temp_voted_y == g_test_y[i] else 0)
            temp_error_correct.append(1 if temp_voted_y_original == g_test_y[i] else 0)
            error_correct_list.append(temp_error_correct)
            voted_y.append(temp_voted_y)
            if isTie:
                tieNumber += 1
                if temp_voted_y == g_test_y[i]:
                    correctTieNumber += 1
            if isTie_original:
                tieNumber_original += 1
                if temp_voted_y_original == g_test_y[i]:
                    correctTieNumber_original += 1
        print("(之前的方法)平票百分比：%s" % (tieNumber_original / len(g_test_y)))
        print("(之前的方法)平票正确百分比：%s" % (correctTieNumber_original / tieNumber_original))
        print("平票百分比：%s" % (tieNumber / len(validate_y)))
        print("平票正确百分比：%s" % (correctTieNumber / tieNumber))
        g_test_f1score = f1_score(g_test_y, voted_y, average='macro')

        g_test_values.append(g_test_f1score)
        output_cell_values(runtime,
                           generation_index,
                           'gtest',
                           g_test_y,
                           voted_y,
                           g_test_f1score,
                           best_dimension_result)
        output_error_correct(runtime,
                             generation_index,
                             'gtest',
                             error_correct_list)

        print("runtime " + str(runtime) +
              " generation "+str(generation_index) +
              ": validate value = "+str(validate_f1score) +
              ": test value = "+str(test_f1score))


        # preparation for next generation
        population = new_population
        parentResults = new_population_result

    validate_results.append(validate_values)
    test_results.append(test_values)
    g_test_results.append(g_test_values)

    # 统计最后一代最优个体三个part的特征频次
    for i in range(split_num):
        for j in range(len(new_population[i][0])):
            if new_population[i][0][j] == 1:
                imp_feature_times[j] += 1
    f = open("./imp_feature_times.csv", "a")
    csv_writer = csv.writer(f, lineterminator="\n")
    csv_writer.writerow(imp_feature_times)
    f.close()

    #
    # plt.plot(validate_values, color="g", label="validate results")
    # plt.plot(test_values, color="r", label="test results")
    # plt.legend()
    # plt.show()
    runtime = runtime + 1

output_results(validate_results,test_results,g_test_results)