import torch
import torch.nn.functional as F
from model import VariationalAutoEncoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def vae_loss(x_recon,x,mean,log_var):
    reconstruction_loss = F.binary_cross_entropy(x_recon,x,reduction='sum')
    kl_div = -0.5* torch.sum(1+log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_div


def train(EPOCHS,model,train_loader,optimizer,device):
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images,_ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            x_recon,mean,log_var = model(images)
            loss = vae_loss(x_recon,images,mean,log_var)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataset)
        print(f"EPOCH:{epoch+1} LOSS:{avg_loss:.4f}")

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    HIDDEN_DIM = 128
    LATENT_DIM = 16
    LR = 1e-3
    EPOCHS = 5

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data",train=True,download=True,transform=transform)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

    model = VariationalAutoEncoder(hidden_dim=HIDDEN_DIM,latent_dim=LATENT_DIM).to(DEVICE) 
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    train(
        EPOCHS=EPOCHS,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=DEVICE
    )

