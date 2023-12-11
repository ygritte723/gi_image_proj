import torch.nn as nn
import torch
# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).




class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            #Input = 3 x 84 x 84, Output = 32 x 84 x 84
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1), 
            nn.ReLU(),
            #Input = 32 x 84 x 84, Output = 32 x 42 x 42
            nn.MaxPool2d(kernel_size=2),
  
            #Input = 32 x 42 x 42, Output = 64 x 42 x 42
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            #Input = 64 x 42 x 42, Output = 64 x 21 x 21
            nn.MaxPool2d(kernel_size=2),
              
            #Input = 64 x 21 x 21, Output = 64 x 21 x 21
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            #Input = 64 x 21 x 21, Output = 64 x 10 x 10
            nn.MaxPool2d(kernel_size=2),
  
            nn.Flatten(),
            nn.Linear(64*10*10, 640*5*5),
            nn.ReLU(),
            
            #nn.Linear(512, 10)
        )
  
    def forward(self, x):
        x = self.model(x)
        x = x.reshape(-1,640,5,5)
        return x

model = Conv()
img = torch.rand((64,3,84,84))
print(model(img).shape)
#print(model)