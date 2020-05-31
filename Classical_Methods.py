# encoding=utf-8
import numpy as np
import cv2, os
import time


def load_img(file_name):
    """
    载入图像，统一尺寸，灰度化处理，直方图均衡化
    :param file_name: 图像文件名
    :return: 图像矩阵
    """
    t_img_mat = cv2.imread(file_name)  # 载入图像
    t_img_mat = cv2.resize(t_img_mat, IMG_SIZE)  # 统一尺寸
    t_img_mat = cv2.cvtColor(t_img_mat, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
    img_mat = cv2.equalizeHist(t_img_mat)  # 直方图均衡
    return img_mat


def create_img_mat(dir_name, algorithm=0):
    """
    生成图像样本矩阵，组织形式为行为属性，列为样本
    :param dir_name: 包含训练数据集的图像文件夹路径
    :param algorithm: 识别算法，0-EigenFace，1-Fisher，2-LBP
    :return: 样本矩阵，标签矩阵
    """
    data_mat = np.zeros((IMG_SIZE[0] * IMG_SIZE[1], 1))
    label = []
    data_list = []
    for parent, dir_names, file_names in os.walk(dir_name):
        for t_dir_name in dir_names:
            # print(t_dir_name, ' in ', dir_names)
            for sub_parent, sub_dir_name, sub_file_names in os.walk(parent + '/' + t_dir_name):
                for t_index, t_file_name in enumerate(sub_file_names):
                    if not t_file_name.endswith('.jpg'):
                        continue
                    if t_file_name.endswith('.10.jpg'):
                        continue
                    t_img_mat = load_img(sub_parent + '/' + t_file_name)
                    img_mat = np.reshape(t_img_mat, (-1, 1))

                    if algorithm == 0:
                        data_mat = np.column_stack((data_mat, img_mat))
                    else:
                        # print(data_mat.shape, '---1---------')
                        # print(img_mat.shape, '---2---------')
                        data_mat = img_mat if t_index == 0 else np.column_stack((data_mat, img_mat))

                    label.append(sub_parent + '/' + t_file_name)
                    # print(data_mat.shape, ":\n", data_mat)
            data_list.append(data_mat[:, 1:] if algorithm == 0 else data_mat)
    return data_mat[:, 1:], label, data_list


def algorithm_pca(data_mat):
    """
    PCA函数，用于数据降维
    :param data_mat: 样本矩阵
    :return: 降维后的样本矩阵和变换矩阵
    """
    mean_mat = np.mat(np.mean(data_mat, 1)).T
    cv2.imwrite('./data/face_test/mean_face.jpg', np.reshape(mean_mat, IMG_SIZE))

    diff_mat = data_mat - mean_mat
    # print('差值矩阵', diff_mat.shape, diff_mat)
    cov_mat = (diff_mat.T * diff_mat) / float(diff_mat.shape[1])
    # print('协方差矩阵', cov_mat.shape, cov_mat)

    eig_vals, eig_vecs = np.linalg.eig(np.mat(cov_mat))
    # print('特征值（所有）：', eig_vals, '特征向量（所有）：', eig_vecs.shape, eig_vecs)

    eig_vecs = diff_mat * eig_vecs
    eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0)
    eig_val = (np.argsort(eig_vals)[::-1])[:DIM]
    eig_vec = eig_vecs[:, eig_val]
    # print('特征值（选取）：', eig_val, '特征向量（选取）：', eig_vec.shape, eig_vec)
    for i in range(0, eig_vec.shape[1]):
        # print('-----',eig_vec.shape)
        # print('-----', eig_vec[:, i].shape)
        # print('-----',np.reshape(eig_vec[:,i], IMG_SIZE).shape)
        pic = np.reshape(eig_vec[:,i], IMG_SIZE)
        # print(pic)
        pic = 10000 * pic
        # print(pic)
        # pic *= (pic > 0)
        # pic = pic * (pic <= 255) + 255 * (pic > 255)
        pic = pic.astype(np.uint8)
        # print(pic)
        cv2.imwrite('./data/face_test/mean_face' + str(i) + '.jpg', pic)
    low_mat = eig_vec.T * diff_mat
    # print('低维矩阵：', low_mat)

    return low_mat, eig_vec


def algorithm_lda(data_list):
    """
    多分类问题的线性判别分析算法
    :param data_list: 样本矩阵列表
    :return: 变换后的矩阵列表和变换矩阵
    """
    n = data_list[0].shape[0]
    Sw = np.zeros((n, n))
    u = np.zeros((n, 1))
    Sb = np.zeros((n, n))
    N = 0
    mean_list = []
    sample_num = []

    for data_mat in data_list:
        mean_mat = np.mat(np.mean(data_mat, 1)).T
        cv2.imwrite('./data/face_test/fisher_face.jpg', np.reshape(mean_mat, IMG_SIZE))
        mean_list.append(mean_mat)
        sample_num.append(data_mat.shape[1])

        data_mat = data_mat - mean_mat
        Sw += data_mat * data_mat.T
    # print('Sw的维度：', Sw.shape)

    for index, mean_mat in enumerate(mean_list):
        m = sample_num[index]
        u += m * mean_mat
        N += m
    u = u / N
    # print('u的维度：', u.shape)

    for index, mean_mat in enumerate(mean_list):
        m = sample_num[index]
        sb = m * (mean_mat - u) * (mean_mat - u).T
        Sb += sb
    # print('Sb的维度：', Sb.shape)

    eig_vals, eig_vecs = np.linalg.eig(np.mat(np.linalg.inv(Sw) * Sb))
    eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0)
    eig_val = (np.argsort(eig_vals)[::-1])[:DIM]
    eig_vec = eig_vecs[:, eig_val]
    # print('选取的特征值：', eig_val.shape)
    # print('变换矩阵维度：', eig_vec.shape)
    for i in range(0, eig_vec.shape[1]):
        # print('-----',eig_vec.shape)
        # print('-----', eig_vec[:, i].shape)
        # print('-----',np.reshape(eig_vec[:,i], IMG_SIZE).shape)
        pic = np.reshape(eig_vec[:,i], IMG_SIZE)
        print(pic)
        pic = 10000 * pic
        print(pic)
        pic *= (pic > 0)
        pic = pic * (pic <= 255) + 255 * (pic > 255)
        pic = pic.astype(np.uint8)
        # print(pic)
        cv2.imwrite('./data/face_test/fisher_face' + str(i) + '.jpg', pic)

    trans_mat_list = []
    for data_mat in data_list:
        trans_mat_list.append(eig_vec.T * data_mat)
    # print('trans_mat_list=', len(trans_mat_list))
    # with open('./data/face_test/class.txt', 'w') as f:
    #     f.write(str(trans_mat_list))
    return trans_mat_list, eig_vec


class AlgorithmLbp(object):
    def __init__(self):
        self.table = {}
        self.ImgSize = IMG_SIZE
        self.BlockNum = 5
        self.count = 0

    def load_img_list(self, dir_name):
        """
        加载图像矩阵列表
        :param dir_name:文件夹路径
        :return: 包含最原始的图像矩阵的列表和标签矩阵
        """
        img_list = []
        label = []
        for parent, dir_names, file_names in os.walk(dir_name):
            for t_dir_name in dir_names:
                for sub_parent, sub_dir_name, sub_filenames in os.walk(parent + '/' + t_dir_name):
                    for file_name in sub_filenames:
                        if not file_name.endswith('.jpg'):
                            continue
                        if file_name.endswith('.10.jpg'):
                            continue
                        img_list.append(load_img(sub_parent + '/' + file_name))  
                        label.append(sub_parent + '/' + file_name)
        return img_list, label

    def get_hop_counter(self, num):
        """
        计算二进制序列是否只变化两次
        :param num: 数字
        :return: 01变化次数
        """
        bin_num = bin(num)
        bin_str = str(bin_num)[2:]
        n = len(bin_str)
        if n < 8:
            bin_str = "0" * (8 - n) + bin_str
        n = len(bin_str)
        counter = 0
        for i in range(n):
            if i != n - 1:
                if bin_str[i + 1] != bin_str[i]:
                    counter += 1
            else:
                if bin_str[0] != bin_str[i]:
                    counter += 1
        return counter

    def get_table(self):
        """
        生成均匀对应字典
        :return: 均匀LBP特征对应字典
        """
        counter = 1
        for i in range(256):
            if self.get_hop_counter(i) <= 2:
                self.table[i] = counter
                counter += 1
            else:
                self.table[i] = 0
        return self.table

    def get_lbp_feature(self, img_mat):
        """
        计算LBP特征
        :param img_mat:图像矩阵
        :return: LBP特征图
        """
        # cv2.imwrite('./data/face_test/lbp_img' + str(self.count) + '.jpg', img_mat)
        m = img_mat.shape[0]
        n = img_mat.shape[1]
        neighbor = [0] * 8
        feature_map = np.mat(np.zeros((m, n)))
        t_map = np.mat(np.zeros((m, n)))
        for y in range(1, m - 1):
            for x in range(1, n - 1):
                neighbor[0] = img_mat[y - 1, x - 1]
                neighbor[1] = img_mat[y - 1, x]
                neighbor[2] = img_mat[y - 1, x + 1]
                neighbor[3] = img_mat[y, x + 1]
                neighbor[4] = img_mat[y + 1, x + 1]
                neighbor[5] = img_mat[y + 1, x]
                neighbor[6] = img_mat[y + 1, x - 1]
                neighbor[7] = img_mat[y, x - 1]
                center = img_mat[y, x]
                temp = 0
                for k in range(8):
                    temp += (neighbor[k] >= center) * (1 << k)
                feature_map[y, x] = self.table[temp]
                t_map[y, x] = temp
        feature_map = feature_map.astype('uint8')  # 数据类型转换为无符号8位型，如不转换则默认为float64位，影响最终效果
        t_map = t_map.astype('uint8')
        # print('t_map', t_map.shape)
        # cv2.imwrite('./data/face_test/lbp_face' + str(self.count) + '.jpg', t_map)
        # print('feature_map', feature_map.shape)
        # cv2.imwrite('./data/face_test/lbp_map' + str(self.count) + '.jpg', feature_map)
        self.count += 1
        return feature_map

    def get_hist(self, roi):
        """
        计算直方图
        :param roi:图像区域
        :return: 直方图矩阵
        """
        hist = cv2.calcHist([roi], [0], None, [59], [0, 256])  # 第四个参数是直方图的横坐标数目，经过均匀化降维后这里一共有59种像素
        return hist

    def compare(self, sampleImg, test_img):
        """
        比较函数，这里使用的是欧氏距离排序，也可以使用KNN，在此处更改
        :param sampleImg: 样本图像矩阵
        :param test_img: 测试图像矩阵
        :return: k2值
        """
        testFeatureMap = self.get_lbp_feature(test_img)
        sampleFeatureMap = self.get_lbp_feature(sampleImg)
        # 计算步长，分割整个图像为小块
        ystep = int(self.ImgSize[0] / self.BlockNum)
        xstep = int(self.ImgSize[1] / self.BlockNum)

        k2 = 0
        for y in range(0, self.ImgSize[0], ystep):
            for x in range(0, self.ImgSize[1], xstep):
                testroi = testFeatureMap[y:y + ystep, x:x + xstep]
                sampleroi = sampleFeatureMap[y:y + ystep, x:x + xstep]
                testHist = self.get_hist(testroi)
                sampleHist = self.get_hist(sampleroi)
                k2 += np.sum((sampleHist - testHist) ** 2) / np.sum((sampleHist + testHist))
        # print('k2的值为', k2)
        return k2

    def predict(self, dir_path, test_path):
        """
        预测函数
        :param dir_path:样本图像文件夹路径
        :param test_path: 测试图像文件路径
        :return: 最相近图像名称
        """
        global acc_count
        self.table = self.get_table()

        test_img_list = []
        test_labels = []
        for parent, dir_names, file_names in os.walk(test_path):
            for t_dir_name in dir_names:
                for sub_parent, sub_dir_name, sub_filenames in os.walk(parent + '/' + t_dir_name):
                    for file_name in sub_filenames:
                        if not file_name.endswith('.jpg'):
                            continue
                        if file_name.endswith('.10.jpg'):
                            test_img_list.append(load_img(sub_parent + '/' + file_name))
                            test_labels.append(sub_parent.split('/')[-1])

        time_1 = int(round(time.time() * 1000))

        img_list, label = self.load_img_list(dir_path)

        time_2 = int(round(time.time() * 1000))
        algorithm_time = time_2 - time_1
        print("Algorithm_LBPH Finished in: ", algorithm_time)
        print("start comparing ......")

        acc_count = 0
        result_list = []
        time_1 = int(round(time.time() * 1000))
        for i in range(0, len(test_img_list)):
            test_img = test_img_list[i]
            k2_list = []
            for img in img_list:
                k2 = self.compare(img, test_img)
                k2_list.append(k2)
            result = label[np.argsort(k2_list)[0]]
            result = result.split('/')[-2]
            result_list.append(result)
            acc_count = acc_count + 1 if result == test_labels[i] else acc_count
            # print(result, '：', test_labels[i])
        time_2 = int(round(time.time() * 1000))
        compare_time = time_2 - time_1

        print("Comparing Finished in: ", compare_time)
        print("---- Accuracy is: ", acc_count / len(result_list), " ----")

        return result_list


def method_compare(mat_list, test_img_vec, algorithm=0):
    """
    比较函数，这里只是用了最简单的欧氏距离比较，还可以使用KNN等方法，如需修改修改此处即可
    :param mat_list: 样本向量集
    :param test_img_vec: 测试图像向量
    :param algorithm: 识别算法，0-EigenFace，1-Fisher，2-LBP
    :return: 与测试图片最相近的图像文件名的index
    """
    dis_list = []
    if algorithm == 0:
        for sample_vec in mat_list.T:
            dis_list.append(np.linalg.norm(test_img_vec - sample_vec))

    elif algorithm == 1:
        # print('mat_list.len=', len(mat_list))
        for trans_mat in mat_list:
            # print('trans_mat.shape=', trans_mat.shape)
            for sample_vec in trans_mat.T:
                dis_list.append(np.linalg.norm(test_img_vec - sample_vec))

    # print('disList=', len(dis_list))
    index = np.argsort(dis_list)[0]
    return index


def method_predict(data_path, test_path, algorithm=0):
    """
    预测函数
    :param data_path: 包含训练数据集的文件夹路径
    :param test_path: 测试图像文件名
    :param algorithm: 识别算法，0-EigenFace，1-Fisher，2-LBP
    :return: 预测结果
    """
    global result_list, acc_list, acc_count, algorithm_time, compare_time

    if algorithm == 2:
        result_list = AlgorithmLbp().predict(data_path, test_path)
    else:
        print("start loading images ......")
        data_mat, label, data_list = create_img_mat(data_path, algorithm)
        print("start calculating ", "Algorithm_PCA" if algorithm == 0 else "Algorithm_LDA", " ......")
        # print('标签信息：', label)
        time_1 = int(round(time.time() * 1000))
        if algorithm == 0:
            mat_list, eig_vec = algorithm_pca(data_mat)
        else:
            mat_list, eig_vec = algorithm_lda(data_list)
        time_2 = int(round(time.time() * 1000))
        algorithm_time = time_2 - time_1
        print("Algorithm_PCA Finished in: " if algorithm == 0 else "Algorithm_LDA Finished in: ", algorithm_time)
        print("start comparing ......")

        test_img_list = []
        test_labels = []
        for parent, dir_names, file_names in os.walk(test_path):
            for t_dir_name in dir_names:
                for sub_parent, sub_dir_name, sub_filenames in os.walk(parent + '/' + t_dir_name):
                    for file_name in sub_filenames:
                        if not file_name.endswith('.jpg'):
                            continue
                        if file_name.endswith('.10.jpg'):
                            test_img_mat = np.reshape(load_img(sub_parent + '/' + file_name), (-1, 1))
                            test_img_vec = np.reshape((eig_vec.T * test_img_mat), (1, -1))
                            test_img_list.append(test_img_vec)
                            test_labels.append(sub_parent.split('/')[-1])

        # print(len(test_img_list), test_img_list)
        # print(len(test_labels), test_labels)

        # test_img_mat = np.reshape(load_img(test_path), (-1, 1))
        # test_img_vec = np.reshape((eig_vec.T * test_img_mat), (1, -1))

        time_1 = int(round(time.time() * 1000))
        result_list = []
        acc_count = 0
        for i in range(0, len(test_img_list)):
            index = method_compare(mat_list, test_img_list[i], algorithm)
            # print(index,":",len(label))
            if index > len(label):
                continue
            result = label[index]
            result = result.split('/')[-2]
            result_list.append(result)
            acc_count = acc_count + 1 if result == test_labels[i] else acc_count
            # print(result, '：', test_labels[i])
        time_2 = int(round(time.time() * 1000))
        compare_time = time_2 - time_1

        print("Comparing Finished in: ", compare_time)
        print("---- Accuracy is: ", acc_count / len(result_list), " ----")
    return result_list


IMG_SIZE = (20, 20)
DIM = 20
if __name__ == '__main__':
    method_predict('./data/Face_Recognition_Data/grimace', './data/Face_Recognition_Data/grimace', 2)
    # method_predict('./data/Face_Recognition_Data/faces94', './data/Face_Recognition_Data/faces94', 0)
    # method_predict('./data/Face_Recognition_Data/faces94', './data/Face_Recognition_Data/faces94', 0)

