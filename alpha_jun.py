# 加载飞桨、NumPy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np


def load_data():
    # 读取文件
    datafile = open('2021.5_BTC_price.txt')
    datafile = datafile.readlines()

    # 提取价格数据到data列表中，同时去除换行符
    data = []
    for i in datafile:
        data.append(eval(i.rstrip().split(' ')[2]))

    # 将列表转化为二维数组，每组都有13个数据
    data = np.array(data, dtype=np.float32)
    data = data[:len(data) // 13 * 13]
    data = data.reshape(len(data) // 13, 13)

    # 计算数据集的最大值，最小值

    maximums, minimums = data.max(axis=0), data.min(axis=0)
    global max_values
    global min_values
    max_values = maximums
    min_values = minimums

    # 记录数据的归一化参数，在预测时对数据做归一化

    for i in range(13):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    return data


class Regressor(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()

        # 定义一层全连接层，输入维度是12，输出维度是1
        self.fc = Linear(in_features=12, out_features=1)

    # 网络的前向计算
    def forward(self, inputs):
        x1 = self.fc(inputs)

        return x1


# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
model.train()
# 加载数据
training_data = load_data()

# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
EPOCH_NUM = 100  # 设置外层循环次数
BATCH_SIZE = 5  # 设置batch大小
losses = []
# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])  # 获得输入数据
        y = np.array(mini_batch[:, -1:])  # 获得预测数据
        x = paddle.to_tensor(x)
        y = paddle.to_tensor(y)
        predicts = model(x)
        # 计算损失
        loss = F.square_error_cost(predicts, label=y)
        avg_loss = paddle.mean(loss)
        losses.append(avg_loss)
        if iter_id % 20 == 0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，以备下一轮计算
        opt.clear_grad()

# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), 'LR_model.pdparams')
print("模型保存成功，模型参数保存在LR_model.pdparams中")

# 参数为保存模型参数的文件地址
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()
# 画出损失函数的变化趋势
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 输入12个连续的5分钟价格，以预测价格作为下一个值，继续组成12个价格，画出走势图
one_data_list = [19205, 19224, 19213, 19218, 19201, 19204, 19241, 19243, 19253, 19275, 19327, 19302]
draw_data = []
for i in range(12):
    one_data = np.array(one_data_list,
                        dtype=np.float32)
    one_data.reshape(1, 12)
    one_data = (one_data - min_values[-1]) / (max_values[-1] - min_values[-1])

    # 将数据转为动态图的variable格式
    one_data = paddle.to_tensor(one_data)
    predict = model(one_data)

    # 对结果做反归一化处理
    predict = predict * (max_values[-1] - min_values[-1]) + min_values[-1]

    one_data_list.pop(0)
    one_data_list.append(predict.numpy().tolist()[0])
    draw_data.append(predict.numpy().tolist()[0])
    print(draw_data)

plot_x = np.arange(len(draw_data))
plot_y = np.array(draw_data)
plt.plot(plot_x, plot_y)
plt.show()
#测试下github在线模式~
