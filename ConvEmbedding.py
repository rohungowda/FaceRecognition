import torch
from Constants import CHANNELS, PATCH_SIZE, ATTENTION_HEADS, N, CNN_EMBEDDING, BATCH_SIZE, CHUNK_SIZE


class ConvEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv_layer_1 =  torch.nn.Conv2d(CHANNELS, int(CNN_EMBEDDING/4), 2, stride=1, padding=0, dtype=torch.float32)
        self.conv_layer_2 =  torch.nn.Conv2d(int(CNN_EMBEDDING/4), int(CNN_EMBEDDING/2) , 2, stride=2, padding=0, dtype=torch.float32)
        self.conv_layer_3 =  torch.nn.Conv2d(int(CNN_EMBEDDING/2), int(CNN_EMBEDDING) , 3, stride=3, padding=0, dtype=torch.float32)
        self.relu_layer = torch.nn.ReLU()
        self.pool_layer = torch.nn.MaxPool2d(2,stride=2)
        

        self.WQ = torch.nn.Parameter(torch.randn((CNN_EMBEDDING,CNN_EMBEDDING), dtype=torch.float32))
        self.WK = torch.nn.Parameter(torch.randn((CNN_EMBEDDING, CNN_EMBEDDING), dtype=torch.float32))
        


    def forward(self, image):
        attention_tensor = torch.empty(image.size(0), ATTENTION_HEADS, int(N), int(N))

        chunks = torch.chunk(image,chunks=(BATCH_SIZE//CHUNK_SIZE),dim=0)


        for b,chunk in enumerate(chunks):
            chunk_reshape = torch.reshape(chunk, (chunk.size(0) * int(N), CHANNELS, PATCH_SIZE, PATCH_SIZE))
            output_1 = self.conv_layer_1(chunk_reshape)
            relu_output_1 = self.relu_layer(output_1)
            pool_output_1 = self.pool_layer(relu_output_1)

            output_2 = self.conv_layer_2(pool_output_1)
            relu_output_2 = self.relu_layer(output_2)
            conv_output_2 = self.pool_layer(relu_output_2)

            output_3 = self.conv_layer_3(conv_output_2)
            embeddings = self.relu_layer(output_3).squeeze(2).squeeze(2)

            division = CNN_EMBEDDING // ATTENTION_HEADS
            Qh = (embeddings @ self.WQ).reshape(-1, ATTENTION_HEADS, int(N),  division)
            Kh = (embeddings @ self.WK).reshape(-1, ATTENTION_HEADS, int(N),  division).permute(0,1,3,2)
            attention_matrix = (Qh @ Kh) / torch.sqrt(torch.tensor(division))

            attention_tensor[(b* attention_matrix.size(0)) : ((b+1)* attention_matrix.size(0)),:,:,:] = attention_matrix

            return attention_tensor