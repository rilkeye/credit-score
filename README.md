# Introduction  
DataFountain Competition - credit score assumption  
website: https://www.datafountain.cn/competitions/337/details/rule  
team ID: 85086  
# requirement
lightgbm : 2.2.3  
pandas : 0.23.4  
numpy : 1.15.4  
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
新增了训练集和验证集的分割功能  
修改了utils.build_data_array函数的返回值  
新增了utils.split_data_and_label函数，用于分割训练集和验证集的data与label  
更新并上传了新的结果
