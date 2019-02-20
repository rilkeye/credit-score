# Introduction  
    DataFountain Competition - credit score assumption  
    website: https://www.datafountain.cn/competitions/337/details/rule  
    team ID: 85086  
# Requirement
    lightgbm : 2.2.3  
    pandas : 0.23.4  
    numpy : 1.15.4  
    sklearn  
# Usage 
    config file is writing...  
    1、python main.py
# Model  
    Contain different models we trained.  
# Data  
    Download from : https://www.datafountain.cn/competitions/337/details/rule  
    include : test_dataset.csv , train_dataset.csv  
    submit_example file is omitted and you can refer Submit.csv in Data.
    result made by us named as Submit.csv , Submit1.csv , Submit2.csv and so on.  

# Update Log
## version 1.0更新时间:2019年2月18日17:30:38  
    首次上传
## version 1.1更新时间:2019年2月18日22:18:44  
    更新了utils.build_data_array函数写入label的bug，并更新新的提交结果Submit.py  
## version 2.0更新时间:2019年2月19日11:40:01  
    新增数据分割功能，分割为训练集，验证集，7:3  
    修改utils.build_data_array函数的返回值  
    新增utils.split_data_and_label函数，用于分割训练集和验证集的data与label  
    更新并上传了新的结果
## version 3.0更新时间:2019年2月19日19:30:55  
    将原来main.py文件模型训练部分移植到train.py，提供handle_data()，train()，make_prediction()接口  
    将数据分割为训练集、验证集、测试集，6:2:2  
    新增通过测试集判断模型性能功能，判断标准参照赛制标准  
    新增训练日志功能，将每次训练参数、训练次数、时间、模型路径和结果保存在'/training log.txt'
## version 3.1更新时间:2019年2月20日22:29:54  
    utils中新增计算特征Pearson相关系数功能  
    计算原有特征的Pearson相关系数，保存在'/data/pearson.txt'
