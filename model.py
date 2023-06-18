import torch as t

class MPL(t.nn.Module):
    def __init__(self):
        super(MPL, self).__init__()
        self.fc1 = t.nn.Linear(64*64*3, 512)
        self.fc2 = t.nn.Linear(512, 100)
        self.fc3 = t.nn.Linear(100, 10)

        # self.flatten = t.nn.Flatten()
        # self.linear_relu_stack = t.nn.Sequential(
        #     t.nn.Linear(64*64*3, 512),
        #     t.nn.ReLU(),
        #     t.nn.Linear(512,512),
        #     t.nn.ReLU(),
        #     t.nn.Linear(512,10),
        # )
    def forward(self, x):
        x = x.view(-1, 64*64*3)
        x = self.fc1(x)
        x = t.relu(x)
        x = self.fc2(x)
        x = t.relu(x)
        x = self.fc3(x)
        return x
        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        # return logits