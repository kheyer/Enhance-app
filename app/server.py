from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO, StringIO
from fastai import *
from fastai.vision import *
import base64
import pdb
from utils import *

model_file_url = 'https://www.dropbox.com/s/vixrjz68hnfvlqx/obj2.pth?raw=1'
model_file_name = 'model'
classes = ['black', 'grizzly', 'teddys']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

class FeatureLoss_Wass(nn.Module):
    def __init__(self):
        super().__init__()
    def make_features(self, x, clone=False):
        return []
    def forward(self, input, target):
        return target.mean()
    def __del__(self): self.hooks.remove()

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def get_resize(y, z, max_size):
    if y*2 <= max_size and z*2 <= max_size:
        y_new = y*2
        z_new = z*2
    else:
        if y > z:
            y_new = max_size
            z_new = int(round_up_to_even(z * max_size / y))

        else:
            z_new = max_size
            y_new = int(round_up_to_even(y * max_size / z))
    return (y_new, z_new)


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    defaults.device = torch.device('cpu')
    data_bunch = (ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(max_zoom=2.), size=224, tfm_y=True)
        .normalize(imagenet_stats, do_y=True))
    data_bunch.c = 3
    arch = models.resnet34
    feat_loss = FeatureLoss_Wass()

    learn = unet_learner(data_bunch, arch, pretrained=False, wd=1e-3, loss_func=feat_loss,
                     blur=True, norm_type=NormType.Weight)

    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

PREDICTION_FILE_SRC = path/'static'/'predictions.txt'
IMG_FILE_SRC = path/'static'/'enhanced_image.png'

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())

    #bytes = base64.b64decode(img_bytes)
    #img = open_image(BytesIO(bytes))

    img = open_image(BytesIO(img_bytes))
    x, y, z = img.data.shape

    max_size = 1000
    y_new, z_new = get_resize(y, z, max_size)

    data_bunch = (ImageImageList.from_folder(path).random_split_by_pct(0.1, seed=42)
          .label_from_func(lambda x: 0)
          .transform(get_transforms(do_flip=False), size=(y_new,z_new), tfm_y=True)
          .databunch(bs=2).normalize(imagenet_stats, do_y=True))

    data_bunch.c = 3
    learn.data = data_bunch
    _,img_hr,losses = learn.predict(img)

    im = Image(img_hr.clamp(0,1))

    im_data = image2np(im.data*255).astype(np.uint8)

    img_io = BytesIO()

    PIL.Image.fromarray(im_data).save(img_io, 'PNG')

    img_io.seek(0)

    img_str = base64.b64encode(img_io.getvalue()).decode()

    #im = Image(img_hr.clamp(0,1))
    #im.save(IMG_FILE_SRC)
    #result_html1 = path/'static'/'result1.html'
    #result_html = str(result_html1.open().read())

    html_out = result_html(img_str)
    return HTMLResponse(html_out)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app = app, host="0.0.0.0", port=8080)
