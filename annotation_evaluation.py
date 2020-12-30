import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy import stats

def fleiss_kappa(M):
    """Computes Fleiss' kappa for group of annotators.

    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix

    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    #print("pj: ", p)
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    #print("Pi: ", P)
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)


def matrixes_generator(col_names, first_question_index, n_videos, categories):
    #VERSION 5 CATEGORIES
    matrix_M = pd.DataFrame([], columns=categories)
    matrix_M1 = pd.DataFrame([], columns=categories)
    for video_i in range(0, n_videos):
        print(" ---> SELECTED QUESTION: ", col_names[first_question_index + (video_i * n_questions)])
        respuestas_videoi = df_annotations[col_names[first_question_index + (video_i * n_questions)]]
        n_annotators = respuestas_videoi.count()
        count_per_category = respuestas_videoi.value_counts().to_dict()
        df_video_row_dflt = pd.DataFrame([[0, 0, 0, 0, 0]], columns=categories)
        for category in count_per_category:
            df_video_row_dflt[category] = count_per_category[category]
        matrix_M = matrix_M.append(df_video_row_dflt)
        matrix_M1=matrix_M.append(df_video_row_dflt)

    # VERSION 3 CATEGORIES
    # colapse Bastante & Mucho [cols 0 & 1]; and Poco & Nada [cols 3 & 4]
    matrix_M_3categories = np.array((matrix_M.values), dtype=int)
    # in col 0 -> Mucha + Basntante
    matrix_M_3categories[:, 0] += matrix_M_3categories[:, 1]
    # in col 1 -> Suficiente
    matrix_M_3categories[:, 1] = matrix_M_3categories[:, 2]
    # in col2 -> Poco + Nada
    matrix_M_3categories[:, 2] = matrix_M_3categories[:, 3] + matrix_M_3categories[:, 4]
    # Remove other columns
    matrix_M_3categories = np.delete(matrix_M_3categories, obj=4, axis=1)
    matrix_M_3categories = np.delete(matrix_M_3categories, obj=3, axis=1)
    
    return matrix_M, matrix_M_3categories
def correlation(matrix_M,matrix_M1):
    nVideos = 10
    for i in range (0,nVideos):
        correlation= pearsonr(matrix_M.values[i], matrix_M1[0].values[i])
        correlation1=stats.spearmanr(matrix_M.values[i], matrix_M1[0].values[i])
        print ("Correlacion Pearson" ,i, correlation)
        print("", i , correlation1)

    return correlation 


if __name__ == '__main__':
    ##### INPUT PARAMTERS #########
    path_annotations = "D:\TFG ANALISIS\Test1CSV.csv"
    first_question_index = 10
     # [6-13] TO DO: IMPLEMENT QUESTION 13?
    second_question_index= 12
    n_questions = 8
    n_videos = 10
    ##### INPUT PARAMTERS #########

    df_annotations = pd.read_csv(path_annotations, sep=",")
    col_names = df_annotations.columns
   # print("col_names", col_names)
    #print("END col_names")
    # matrix_M -> rows: videos, cols: categories
    categories = ["Mucho", "Bastante", "Suficiente", "Poco", "Nada"]
    stress_categories = ["Muy estresada", "Estresada", "Normal", "Relajada", "Muy relajada"]
    if(first_question_index==9):
        matrix_M, matrix_M_3categories = matrixes_generator(col_names, first_question_index, n_videos, stress_categories)
    else:
        matrix_M, matrix_M_3categories = matrixes_generator(col_names, first_question_index, n_videos, categories)
        matrix_M1 = matrixes_generator(col_names, second_question_index, n_videos, categories)
    #DEBUG FLEISS KAPPA
    # zero_matrix = np.zeros((n_videos, 5), dtype=int)
    # zero_matrix[0,0] = 14
    # zero_matrix[1,1] = 14
    # zero_matrix[2,2] = 14
    # zero_matrix[3,3] = 14
    # zero_matrix[4,4] = 14
    # zero_matrix[5,0] = 14
    # zero_matrix[6,1] = 14
    # zero_matrix[7,2] = 14
    # zero_matrix[8,3] = 14
    # zero_matrix[9,4] = 14
    # fKappa = fleiss_kappa(zero_matrix)
    print("valores")
    print(matrix_M1[0].values)
    fKappa = fleiss_kappa(matrix_M.values)
    print("Fleiss Kappa (5 categories: Mucho, Bastante, Suficiente, Poco, Nada): ", str(fKappa))
    fKappa_3categories = fleiss_kappa(matrix_M_3categories)
    print("Fleiss Kappa (3 categories: Bastante, Sufi, Poco): ", str(fKappa_3categories))
    print("Values2", matrix_M1[0].values[4])
    print("Values1" ,matrix_M.values[4] )
    corref = correlation(matrix_M,matrix_M1)
    print ("Correlacion a 5 valores " , corref)

