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
### version 1.1更新时间:2019年2月18日22:18:44  
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
### version 3.1更新时间:2019年2月20日22:29:54  
    utils中新增计算特征Pearson相关系数功能  
    计算原有特征的Pearson相关系数，保存在'/data/pearson.txt'  
### version 3.2更新时间2019年2月21日20:40:08
    '/data/MI.txt'为原特征的互信息  
    对模型参数进行了调整，调参方法见：https://www.imooc.com/article/43784?block_id=tuijian_wz  
    对特征进行了以下处理：1，将'当月是否景点游览'、'是否经常逛商场的人'、'当月是否逛过福州仓山万达'、'当月是否到过福州山姆会员店'、  
    '当月是否看电影'、'当月是否景点游览'、'当月是否体育场馆消费'的特征加合，构成新特征“进行过多少种高端消费（max：7）”，并将原来  
    对应的特征删除。  
### version 3.3更新时间2019年2月22日14:39:58  
    新增visualization.py，process_dataframe.py文件，代码取于赵志舜的.ipynb jupyter notebook文件  
    现做特征工程只需要在process_dataframe.py中更改函数中的内容进行即可  
## version 4.0更新时间
    新增main2.py文件，可直接运行利用K折检验法分割数据和做交叉验证。直接运行可以获得预测结果。此法耗时较长，精度高，适合用于作提交结果的预测。  
    main.py文件与main2.py相互独立，main与main2共享train、processed_dataframe、utils等函数，main耗时短但精度低，设置可预测得分的测试集，  
    适合用于平时调整特征工程时使用。  
在此特别感谢[Venn](https://github.com/wangvenn/Credit-Scoring-Regression)的baseline开源。
## version 5.0更新时间2019年2月27日07:26:40
    新增main3.py文件，负责模型ensemble实现，框架已搭建尚未调试，线上线下得分均不足
    main2文件已更新为lgb+xgb的bagging，权重为0.3，0.3，0.4，且作参数调整，线上评分0.06386029000，作为新阶段baseline。（该训练耗时极长）
    CVfunction_parameters_setting.py；para_setting分别为cv函数调参法和gridsearch调参法，目前这两种方法实际调参效果无效，原因不明。
### version 5.1更新时间2019年3月1日22:33:33
    修正了train.py文件中lgb框架的使用方法，lgb_para_setting.py函数调参文件已经可用并验证有效（使用方法询问yzc）  
    已完成lgb的粗调参  
    为特征新增了one-hot编码功能，并增加了4类one-hot特征  
    目前main2效果为best，线上评分0.06388384000
