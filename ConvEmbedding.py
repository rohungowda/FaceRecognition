import torch
from Constants import EMBEDDING_DIM, IMAGE_SIZE, CHANNELS, PATCH_SIZE


class ConvEmbeddings(torch.nn.Module):
    def __init__(self, position_embed):
        super().__init__()
    
    def forward(self, image):
        pass


image = torch.randn((128*64,CHANNELS,int(PATCH_SIZE), int(PATCH_SIZE)))
#pool_layer = torch.nn.MaxPool2d(3, stride=3)
e= 256

conv_layer =  torch.nn.Conv2d(CHANNELS, int(e/8), 2, stride=1, padding=0)
conv_layer_1 =  torch.nn.Conv2d(int(e/8), int(e/6) , 2, stride=1, padding=0)
# relu
# maxpool
conv_layer_2 =  torch.nn.Conv2d(int(e/6), int(e/4) , 3, stride=3, padding=0)
conv_layer_3 =  torch.nn.Conv2d(int(e/4), int(e/2), 3, stride=3, padding=0)
# relu
# maxpool
conv_layer_4 =  torch.nn.Conv2d(int(e/2), int(e), 3, stride=3, padding=0)

output = conv_layer(image)
print(output.size())
output = conv_layer_1(output)
print(output.size())
output = conv_layer_2(output)
print(output.size())
output = conv_layer_3(output)
print(output.size())
output = conv_layer_4(output).squeeze(3).squeeze(2).reshape(64,64,-1)
print(output.size())

# run regular attention

WQ = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float64))
WK = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float64))
WV = torch.nn.Parameter(torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), dtype=torch.float64))


#output = pool_layer(output)
#output = torch.reshape(output,(-1,output.size(2) * output.size(3), EMBEDDING_DIM))
#print(output.size())

#output = pool_layer(output)
#output = torch.reshape(output,(-1,output.size(2) * output.size(3), EMBEDDING_DIM))
#print(output.size())