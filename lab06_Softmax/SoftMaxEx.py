import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
#Low-Level Cross Entropy Loss
x_train = [[1,2,1,1],
           [2,1,3,2],
           [3,1,3,4],
           [4,1,5,5],
           [1,7,5,5],
           [1,2,5,6],
           [1,6,6,6],
           [1,7,7,7]]
y_train = [2,2,2,1,1,1,0,0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
# #Low-Level Cross entropy
# for epoch in range(nb_epochs + 1):
#
#     # Cost
#     hypothesis = F.softmax(x_train.matmul(W) + b, dim=1) # or .mm or @
#     y_one_hot = torch.zeros_like(hypothesis)
#     y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
#     cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     # 100번마다 로그 출력
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))
#
# #F.cross_entropy
# for epoch in range(nb_epochs + 1):
#
#     # Cost 계산 (2)
#     z = x_train.matmul(W) + b # or .mm or @
#     cost = F.cross_entropy(z, y_train)
#
#     # cost로 H(x) 개선
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     # 100번마다 로그 출력
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))
#
#

#High-level Implementation with nn.Module
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
