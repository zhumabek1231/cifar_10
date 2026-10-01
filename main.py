from fastapi import FastAPI,UploadFile,File,HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
classes = ['airplane',
           'automobile',
           'bird',
           'cat',
           'deer',
           'dog',
           'frog',
           'horse',
           'ship',
           'truck'
           ]


class CifarClassifaction(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, image):
        image = self.first(image)
        image = self.second(image)
        return image

transforms =transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()

])


image_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CifarClassifaction()
model.load_state_dict(torch.load('model1_path',map_location= device))
model.to(device)
model.eval()

@image_app.post ('/predict/')
async def check_image (file: UploadFile=File(...)):
     try:
         image_data = await file. read()
         if not image_data:
             raise HTTPException (status_code=400, detail= 'Файл кошулган жок')
         img = Image.open (io.BytesIO (image_data))
         img_tenser = transforms(img).unsqueeze(0).to(device)

         with torch.no_grad():
             y_pred = model(img_tenser)
             pred = y_pred.argmax(dim= 1).item()
         return {"Answer": classes[pred]}

     except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__=='__main__':
    uvicorn.run(image_app, host='127.0.0.1', port=8000)