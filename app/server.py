import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/open?id=1-KKH1BbpIS0-zkQ4LOtSePBFey3UWthT'
export_file_name = 'car_final_resnet50.pth'

classes = ['acura_cl_1997', 'acura_cl_1998', 'acura_cl_1999', 'acura_cl_2001', 'acura_cl_2002', 'acura_cl_2003', 'acura_el_1997', 'acura_el_2001', 'acura_el_2003', 'acura_ilx_2013', 'acura_ilx_2014', 'acura_integra_1900', 'acura_integra_1986', 'acura_integra_1987', 'acura_integra_1988', 'acura_integra_1989', 'acura_integra_1990', 'acura_integra_1991', 'acura_integra_1992', 'acura_integra_1993', 'acura_integra_1994', 'acura_integra_1995', 'acura_integra_1996', 'acura_integra_1997', 'acura_integra_1998', 'acura_integra_1999', 'acura_integra_2000', 'acura_integra_2001', 'acura_kang_2007', 'acura_legend_1987', 'acura_legend_1988', 'acura_legend_1989', 'acura_legend_1990', 'acura_legend_1991', 'acura_legend_1992', 'acura_legend_1993', 'acura_legend_1994', 'acura_legend_1995', 'acura_legend_1996', 'acura_legend_2001', 'acura_legend_2002', 'acura_mdx_1900', 'acura_mdx_1996', 'acura_mdx_2001', 'acura_mdx_2002', 'acura_mdx_2003', 'acura_mdx_2004', 'acura_mdx_2005', 'acura_mdx_2006', 'acura_mdx_2007', 'acura_mdx_2008', 'acura_mdx_2009', 'acura_mdx_2010', 'acura_mdx_2011', 'acura_mdx_2012', 'acura_mdx_2014', 'acura_mdx_2016', 'acura_nsx_2002', 'acura_rdx_2007', 'acura_rdx_2008', 'acura_rdx_2009', 'acura_rdx_2010', 'acura_rdx_2011', 'acura_rdx_2012', 'acura_rdx_2013', 'acura_rdx_2014', 'audi_a4_1998']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
