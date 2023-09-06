import scanpy as sc
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import scipy.stats as stats
import re
import json
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse

def direction(predmeans,ctrlmeans,realmeans,genede_idx):
    predmeans =predmeans[genede_idx]
    ctrlmeans = ctrlmeans[genede_idx]
    realmeans = realmeans[genede_idx]

    preddirection = np.sign(predmeans - ctrlmeans)

    realdirection = np.sign(realmeans -ctrlmeans)
    consistent_signs = (preddirection == realdirection)
    direction = np.count_nonzero(consistent_signs) / len(genede_idx)
    return 1-direction
def evaluation_direction(adata,pertlist,genepred):

    # 例：
    #pertlist=['IER3IP1+ctrl', 'DAD1+ctrl', 'AMIGO3+ctrl', 'DARS+ctrl', 'SYVN1+ctrl', 'GMPPB+ctrl', 'CAD+ctrl', 'GNPNAT1+ctrl', 'DERL2+ctrl', 'FARSB+ctrl', 'TARS+ctrl', 'TTI1+ctrl', 'TIMM44+ctrl',
    # 'PTDSS1+ctrl', 'DDOST+ctrl', 'CHERP+ctrl', 'COPZ1+ctrl', 'DNAJC19+ctrl', 'XRN1+ctrl', 'SPCS2+ctrl']
    result={}
    ctrlreal = adata[adata.obs.condition == 'ctrl'].X.toarray() # type: ignore
    for i in tqdm(pertlist, desc="Processing"):
        metic={}
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
        genede_idx20 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_20'][f'K562(?)_{i}_1+1']
            ]
        genede_idx50 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_50'][f'K562(?)_{i}_1+1']
            ]
        genede_idx100 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_100'][f'K562(?)_{i}_1+1']
            ]
        genede_idx200 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_200'][f'K562(?)_{i}_1+1']
            ]
        genede_idxall = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_all'][f'K562(?)_{i}_1+1']
            ]
        real = adata[adata.obs.condition == i].X.toarray() # type: ignore
        predmeans = np.mean(genepred, axis=0)
        realmeans = np.mean(real, axis=0)
        ctrlmeans = np.mean(ctrlreal, axis=0)
        metic['directionall']=direction(predmeans, ctrlmeans, realmeans, list(range(len(ctrlmeans))))
        metic['directiondeall']=direction(predmeans, ctrlmeans, realmeans, genede_idxall)
        metic['direction20']=direction(predmeans, ctrlmeans, realmeans, genede_idx20)
        metic['direction50']=direction(predmeans, ctrlmeans, realmeans, genede_idx50)
        metic['direction100']=direction(predmeans, ctrlmeans, realmeans, genede_idx100)
        metic['direction200']=direction(predmeans, ctrlmeans, realmeans, genede_idx200)
        result[i]= metic
    return result

def jacard(de_genesid,genede_idx):
    preddeidx = np.array(de_genesid)[:len(genede_idx)]
    intersection = np.intersect1d(preddeidx, np.array(genede_idx))
    return len(intersection)/(len(de_genesid) + len(genede_idx) - len(intersection))
def evaluation_jacard(adata,pertlist,genepred):
    result={}
    ctrl_mean = np.array(adata[adata.obs["condition"] == "ctrl"].to_df().mean().values)

    for i in tqdm(pertlist, desc="Processing"):
        metic={}
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
        genede_idx20 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_20'][f'K562(?)_{i}_1+1']
            ]
        genede_idx50 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_50'][f'K562(?)_{i}_1+1']
            ]
        genede_idx100 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_100'][f'K562(?)_{i}_1+1']
            ]
        genede_idx200 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_200'][f'K562(?)_{i}_1+1']
            ]
        genede_idxall = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_all'][f'K562(?)_{i}_1+1']
            ]
        adataY = adata[adata.obs['condition'] == 'ctrl']
        pert_mean = np.array(adata[adata.obs["condition"] == i].to_df().mean().values)
        de_value = pert_mean-ctrl_mean
        de_value = np.absolute(de_value)
        sorted_deid = sorted(range(len(de_value)), key = lambda x:de_value[x],reverse=True)
        #groups = result['names'].dtype.names  # 獲取分組的名稱
        stat,p_value = stats.mannwhitneyu(gene1pred,adataY.X.toarray(),alternative='less') # type: ignore

        non_zero_cols = np.where(np.count_nonzero(genepred, axis=0) >= (genepred.shape[0] / 10))[0]
        de_genesid=[]
        for item in sorted_deid:
            if p_value[item]<=0.05 and item in non_zero_cols:
                de_genesid.append(item)

        metic['jacarddeall']=jacard(de_genesid, genede_idxall)
        metic['jacardde20']=jacard(de_genesid, genede_idx20)
        metic['jacardde50']=jacard(de_genesid, genede_idx50)
        metic['jacardde100']=jacard(de_genesid, genede_idx100)
        metic['jacardde200']=jacard(de_genesid, genede_idx200)
        
        result[i]= metic
    return result
def evaluation_normMSE(adata,pertlist,genepred):
    result={}
    ctrlmeans = np.mean(adata[adata.obs.condition == 'ctrl'].X.toarray() ,axis=0)# type: ignore
    for i in tqdm(pertlist, desc="Processing"):
        metic={}
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
        genede_idx20 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_20'][f'K562(?)_{i}_1+1']
            ]
        genede_idx50 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_50'][f'K562(?)_{i}_1+1']
            ]
        genede_idx100 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_100'][f'K562(?)_{i}_1+1']
            ]
        genede_idx200 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_200'][f'K562(?)_{i}_1+1']
            ]
        genede_idxall = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_all'][f'K562(?)_{i}_1+1']
            ]

        predmeans = np.mean(np.array(genepred)  ,axis=0)# type: ignore
        
        realmeans = np.mean(adata[adata.obs.condition == i].X.toarray()  ,axis=0)# type: ignore

        noperturb_mse = mse(realmeans,ctrlmeans)
        noperturb_mse20 = mse(realmeans[genede_idx20],ctrlmeans[genede_idx20])
        noperturb_mse50 = mse(realmeans[genede_idx50],ctrlmeans[genede_idx50])
        noperturb_mse100 = mse(realmeans[genede_idx100],ctrlmeans[genede_idx100])
        noperturb_mse200 = mse(realmeans[genede_idx200],ctrlmeans[genede_idx200])
        noperturb_mseall = mse(realmeans[genede_idxall],ctrlmeans[genede_idxall])

        pred_mse = mse(realmeans,predmeans)
        pred_mse20 = mse(realmeans[genede_idx20],predmeans[genede_idx20])
        pred_mse50 = mse(realmeans[genede_idx50],predmeans[genede_idx50])
        pred_mse100 = mse(realmeans[genede_idx100],predmeans[genede_idx100])
        pred_mse200 = mse(realmeans[genede_idx200],predmeans[genede_idx200])
        pred_mseall = mse(realmeans[genede_idxall],predmeans[genede_idxall])

        metic['noperturb_mse']=noperturb_mse
        metic['noperturb_mse20']=noperturb_mse20
        metic['noperturb_mse50']=noperturb_mse50
        metic['noperturb_mse100']=noperturb_mse100
        metic['noperturb_mse200']=noperturb_mse200
        metic['noperturb_mseall']=noperturb_mseall

        metic['pred_mse']=pred_mse
        metic['pred_mse20']=pred_mse20
        metic['pred_mse50']=pred_mse50
        metic['pred_mse100']=pred_mse100
        metic['pred_mse200']=pred_mse200
        metic['pred_mseall']=pred_mseall
        result[i] = metic
    return result

def perturb_variation(data1, data2,geneidx):
    # 示例数据
    #计算每列特征在data_set1中的值是否在data_set2的上下四分位点范围内
    data1 = data1[:,geneidx]
    data2 = data2[:,geneidx]
    iqr_data1 = stats.iqr(data1, axis=0)
    q1_data2, q3_data2 = np.percentile(data2, [25, 75], axis=0)
    
    upper_limit_data2 = q3_data2 + 1.5 * iqr_data1
    lower_limit_data2 = q1_data2 - 1.5 * iqr_data1
    
    within_range = np.logical_and(data1 >= lower_limit_data2, data1 <= upper_limit_data2)
    iqr_ratio = np.mean(within_range, axis=0)
    return np.mean(iqr_ratio)

def evaluation_perturb_variation(adata,pertlist,genepred):
    result={}
    
    for i in tqdm(pertlist, desc="Processing"):
        metic={}
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
        genede_idx20 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_20'][f'K562(?)_{i}_1+1']
            ]
        genede_idx50 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_50'][f'K562(?)_{i}_1+1']
            ]
        genede_idx100 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_100'][f'K562(?)_{i}_1+1']
            ]
        genede_idx200 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_200'][f'K562(?)_{i}_1+1']
            ]
        genede_idxall = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_all'][f'K562(?)_{i}_1+1']
            ]

       
        real = adata[adata.obs.condition == i].X.toarray() # type: ignore
        metic['pert_variationall']=perturb_variation(genepred, real, list(range(real.shape[1])))
        metic['pert_variationdeall']=perturb_variation(genepred, real, genede_idxall)
        metic['pert_variation20']=perturb_variation(genepred, real, genede_idx20)
        metic['pert_variation50']=perturb_variation(genepred, real,genede_idx50)
        metic['pert_variation100']=perturb_variation(genepred, real, genede_idx100)
        metic['pert_variation200']=perturb_variation(genepred, real, genede_idx200)

        

        result[i]= metic
    return result



def STD(data1, data2,geneidx):
    # 示例数据
    data1 = data1[:,geneidx]
    data2 = data2[:,geneidx]

    # 计算均值和标准差
    mean_data1 = np.mean(data1, axis=0)
    std_data2 = np.std(data2, axis=0)
    mean_data2 = np.mean(data2, axis=0)

    # 计算上下界
    upper_bound_data2 = mean_data2 + std_data2
    lower_bound_data2 = mean_data2 - std_data2

    # 判断均值是否在标准差范围内，计算满足条件的列的比例
    within_range = np.logical_and(mean_data1 >= lower_bound_data2, mean_data1 <= upper_bound_data2)
    proportion_within_range = np.sum(within_range) / len(mean_data1)

    return  proportion_within_range

def evaluation_STD(adata,pertlist,genepred):
    result={}
    for i in tqdm(pertlist, desc="Processing"):
        metic={}
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
        genede_idx20 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_20'][f'K562(?)_{i}_1+1']
            ]
        genede_idx50 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_50'][f'K562(?)_{i}_1+1']
            ]
        genede_idx100 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_100'][f'K562(?)_{i}_1+1']
            ]
        genede_idx200 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_200'][f'K562(?)_{i}_1+1']
            ]
        genede_idxall = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_all'][f'K562(?)_{i}_1+1']
            ]

       
        real = adata[adata.obs.condition == i].X.toarray() # type: ignore
    
        metic['STDall']=STD(genepred, real, list(range(real.shape[1])))
        metic['STDdeall']=STD(genepred, real, genede_idxall)
        metic['STD20']=STD(genepred, real, genede_idx20)
        metic['STD50']=STD(genepred, real,genede_idx50)
        metic['STD100']=STD(genepred, real, genede_idx100)
        metic['STD200']=STD(genepred, real, genede_idx200)

        

        result[i]= metic
    return result



def Zscore(predicted_expression, true_expression,geneidx):
    # 示例数据

    # 计算均值和标准差
    z_scores = (predicted_expression.mean(axis=0) - true_expression.mean(axis=0)) / true_expression.std(axis=0)

    # 选取前20个差异表达基因的 z 分数
    de_genes_z_scores = z_scores[geneidx]

    # 计算每个基因的 z 分数均值
    mean_z_scores = np.mean(de_genes_z_scores, axis=0)
  

    return   mean_z_scores

def evaluation_Zscore(adata,pertlist,genepred):
    result={}


    for i in tqdm(pertlist, desc="Processing"):
        metic={}
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
        genede_idx20 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_20'][f'K562(?)_{i}_1+1']
            ]
        genede_idx50 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_50'][f'K562(?)_{i}_1+1']
            ]
        genede_idx100 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_100'][f'K562(?)_{i}_1+1']
            ]
        genede_idx200 = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_200'][f'K562(?)_{i}_1+1']
            ]
        genede_idxall = [
                gene2idx[gene_raw2id[i]]
                for i in adata.uns['top_non_dropout_de_all'][f'K562(?)_{i}_1+1']
            ]
        real = adata[adata.obs.condition == i].X.toarray() # type: ignore
    
        metic['Zscoreall']=Zscore(genepred, real, list(range(real.shape[1])))
        metic['Zscoredeall']=Zscore(genepred, real, genede_idxall)
        metic['Zscore20']=Zscore(genepred, real, genede_idx20)
        metic['Zscore50']=Zscore(genepred, real,genede_idx50)
        metic['Zscore100']=Zscore(genepred, real, genede_idx100)
        metic['Zscore200']=Zscore(genepred, real, genede_idx200)

        

        result[i]= metic
    return result


def gene2sim(adata,gene1,gene2,gene1pred,gene2pred):
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
    gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
    gene1de_idx = [
            gene2idx[gene_raw2id[i]]
            for i in adata.uns['rank_genes_groups_cov_all'][f'K562(?)_{gene1}_1+1']
        ]
    gene2de_idx = [
            gene2idx[gene_raw2id[i]]
            for i in adata.uns['rank_genes_groups_cov_all'][f'K562(?)_{gene2}_1+1']
        ]
    union_set = set(gene1de_idx) | set(gene2de_idx)
    # 将并集转换回列表
    union_list = list(union_set)
    gene1means = np.mean(gene1pred[:,union_list], axis=0)

    gene2means = np.mean(gene2pred[:,union_list], axis=0)

    ctrlreal = adata[adata.obs.condition == 'ctrl'].X.toarray()
    ctrlmeans = np.mean(ctrlreal[:,union_list], axis=0)
    

    pearson_corr = np.corrcoef(gene1means-ctrlmeans,gene2means-ctrlmeans)[0, 1]
    return pearson_corr


