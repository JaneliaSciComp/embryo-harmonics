# %%
import logging
import os
import time

from tqdm import tqdm

from embryoharmonics import celegans, io
from embryoharmonics import Harmonics

# %%
# Set up logging
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("embryoharmonics")
logger.setLevel(logging.INFO)
timestr = time.strftime("%Y%m%d-%H%M%S")
handler = logging.FileHandler(os.path.join(log_dir, f'process_all_geometries_{timestr}.log'))
formatter = logging.Formatter('[%(asctime)s - %(name)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%
PATH = "<your_path_to_celegans_models.h5>"
N = 300
embryo_model_loader = celegans.EmbryoModelLoader(PATH)
root = os.path.normpath(os.path.join(os.getcwd(), 'results'))
os.makedirs(root, exist_ok=True)

# %%
for time_step in tqdm(embryo_model_loader.time_steps):
    logger.info("Processing time step %d", time_step)
    embryo_model = embryo_model_loader.load(time_step)
    mesh = embryo_model.generate_mesh(mesh_size=5)
    harmonics = Harmonics.compute(mesh, n=N)

    mesh_path = os.path.join(root, f"data_{time_step:03d}.vtu")
    logger.info("Saving mesh to %s", mesh_path)
    io.save_mesh(mesh_path, mesh)

    harmonics_path = os.path.join(root, f"harmonics_{time_step:03d}.h5")
    logger.info("Saving harmonics to %s", harmonics_path)
    io.save_harmonics(harmonics_path, harmonics)

# %%
