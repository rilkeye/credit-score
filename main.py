import utils
import lightgbm as lgb

'''
载入train_dataset.csv、test_dataset.csv文件，构建二维数组
缺省train、test首列：“用户编码”；train末列：“信用分”
'''
train_dataset_path = 'data/train_dataset.csv'
test_dataset_path = 'data/test_dataset.csv'

# data_array: shape = (50000,28); score: shape = (50000,)
train_data_array, score = utils.build_data_array(train_dataset_path, tag='train')
test_data_array = utils.build_data_array(test_dataset_path, tag='test')


# 将numpy数组加载到数据集中
train_data = lgb.Dataset(train_data_array, label=score)
train_data.save_binary('data/train.bin')

# Parameters setting
num_round = 1
param = {'objective':'regression'}
param['metric'] = 'auc'

# Train & save model as model.txt
bst = lgb.train(params=param, train_set=train_data, num_boost_round=num_round, valid_sets=None)
bst.save_model('model/model.txt')

# Load model and make prediction
bst = lgb.Booster(model_file='model/model.txt')
pred = bst.predict(test_data_array)
pred = [int(round(score)) for score in pred] # 将pred中的float型四舍五入

# 将结果按赛制要求写入文件
utils.write_SubmitionFile(pred, test_dataset_path)
