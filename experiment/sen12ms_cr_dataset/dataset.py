import logging
import os
import time
from enum import Enum
from glob import glob
import numpy as np
from numpy import array
from torch.utils.data import Dataset
from typing import Tuple, List, Union
# Geographic information systems use GeoTIFF and other formats to organize and store gridded raster datasets
# such as satellite imagery and terrain models. Rasterio reads and writes these formats and provides a Python
# API based on Numpy N-dimensional arrays and GeoJSON.
import rasterio
from rasterio.coords import BoundingBox


class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []


class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = []


class Season(Enum):
    SPRING = 'ROIs1158_spring'
    SUMMER = 'ROIs1868_summer'
    FALL = 'ROIs1970_fall'
    WINTER = 'ROIs2017_winter'
    ALL = [SPRING, SUMMER, FALL, WINTER]


class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    s2cloudy = "s2_cloudy"


class SEN12MSCRTriplet(object):

    SUPPORTED_EXTENSIONS = ['tif', 'npy', 'npz']

    def __init__(self, dataset_dir: str, season: Season, scene_id: str, patch_id: str, file_extension: str):
        self.dataset_dir = dataset_dir
        self.season = season
        self.scene_id = scene_id
        self.patch_id = patch_id
        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f'file_extension:{file_extension} is not supported. Only support:{self.SUPPORTED_EXTENSIONS}')
        self.file_extension = file_extension

    @property
    def data(self) -> Tuple[np.array, np.array, np.array]:
        """ Returns a triplet of orresponding Sentinel 1,
        Sentinel 2 and cloudy Sentinel 2 data.
        Array Shape is CHW.
        """
        s1 = self._load(self.s1_path(), S1Bands.ALL)
        s2 = self._load(self.s2_path(), S2Bands.ALL)
        s2_cloudy = self._load(self.s2_cloudy_path(), S2Bands.ALL)
        return s1, s2, s2_cloudy

    def s1_path(self, dataset_dir: str = None, file_extension: str = None) -> str:
        if not dataset_dir:
            dataset_dir = self.dataset_dir
        if not file_extension:
            file_extension = self.file_extension
        season_dirname = f'{self.season.value}_s1'
        scene_dirname = f's1_{self.scene_id}'
        filename = f'{self.season.value}_s1_{self.scene_id}_p{self.patch_id}.{file_extension}'
        return os.path.join(dataset_dir, season_dirname, scene_dirname, filename)

    def s2_path(self, dataset_dir: str = None, file_extension: str = None) -> str:
        if not dataset_dir:
            dataset_dir = self.dataset_dir
        if not file_extension:
            file_extension = self.file_extension
        season_dirname = f'{self.season.value}_s2'
        scene_dirname = f's2_{self.scene_id}'
        filename = f'{self.season.value}_s2_{self.scene_id}_p{self.patch_id}.{file_extension}'
        return os.path.join(dataset_dir, season_dirname, scene_dirname, filename)

    def s2_cloudy_path(self, dataset_dir: str = None, file_extension: str = None) -> str:
        if not dataset_dir:
            dataset_dir = self.dataset_dir
        if not file_extension:
            file_extension = self.file_extension
        season_dirname = f'{self.season.value}_s2_cloudy'
        scene_dirname = f's2_cloudy_{self.scene_id}'
        filename = f'{self.season.value}_s2_cloudy_{self.scene_id}_p{self.patch_id}.{file_extension}'
        return os.path.join(dataset_dir, season_dirname, scene_dirname, filename)

    def _load(self, path: str, band: Union[S1Bands, S2Bands]) -> np.array:
        if self.file_extension == 'tif':
            return self._load_tif(path, band)
        elif self.file_extension == 'npy':
            return self._load_npy(path)
        elif self.file_extension == 'npz':
            return self._load_npz(path)
        else:
            raise ValueError(f'Data loading error. self.file_extension:{self.file_extension} is not supported.')

    def _load_tif(self, patch_path: str, band: Union[S1Bands, S2Bands]) -> np.array:
        band = band.value
        with rasterio.open(patch_path) as patch:
            data = patch.read(band)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        if len(data.shape) != 3:
            raise ValueError(f'patch:{patch_path} load error. Patch Shape is:{data.shape}')
        return data

    def _load_npy(self, patch_path: str) -> np.array:
        with open(patch_path, 'rb') as f:
            return np.load(f)

    def _load_npz(self, patch_path: str) -> np.array:
        raise NotImplementedError(f'_load_npz hasn\'t been implemented!')

    def _save_patch(self, patch_path: str, patch: np.array):
        if not os.path.exists(os.path.dirname(patch_path)):
            os.makedirs(os.path.dirname(patch_path))
        if os.path.exists(patch_path):
            raise ValueError(f'patch:{patch_path} has existed')
        with open(patch_path, 'wb') as f:
            np.save(f, patch)

    def save(self, dataset_dir: str, data: Tuple[np.array, np.array, np.array] = None) -> None:
        if not data:
            data = self.data
        s1, s2, s2_cloudy = data
        s1_path = self.s1_path(dataset_dir, file_extension='npy')
        s2_path = self.s2_path(dataset_dir, file_extension='npy')
        s2_cloudy_path = self.s2_cloudy_path(dataset_dir, file_extension='npy')
        self._save_patch(s1_path, s1)
        self._save_patch(s2_path, s2)
        self._save_patch(s2_cloudy_path, s2_cloudy)


class SEN12MSCRDataset(Dataset):
    """ Generic data loading routines for the SEN12MS-CR dataset of corresponding Sentinel 1,
    Sentinel 2 and cloudy Sentinel 2 data.

    The files in base_dir shoule be organized as:
    ├── SEN12MS_CR                                      # This is your base dir name
    │   ├── ROIs1868_summer_s1                          # Subdir should be named as {seasons.value}_{sensor.value}
    │   │   ├── s1_102                                  # There should be named as  {sensor.value}_{scene_id}
    │   │   │   │   ├── ROIs1868_summer_s1_102_p100.tif # Named as  {seasons.value}_{sensor.value}_{scene_id}_{patch_id}.tif
    │   │   │   │   ├── ...
    │   │   ├── ...
    │   ├── ...
    Note: The order in which you request the bands is the same order they will be returned in.
    """

    def __init__(self, base_dir, file_extension: str = 'tif'):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            raise Exception(f'SEN12MSCRDataset faled to init. base_dir:{base_dir} does not exist')
        if file_extension not in SEN12MSCRTriplet.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f'file_extension:{file_extension} is not supported. Only support:{SEN12MSCRTriplet.SUPPORTED_EXTENSIONS}'
            )
        self.file_extension = file_extension
        self.triplets = self.get_all_triplets()
        # FOR DEBUG
        # self.triplets = self.triplets[:2048]

    def get_scene_ids(self, season: Season) -> List[str]:
        """ Returns a list of scene ids for a specific season.
        """
        sensor = Sensor('s1')
        season_dirname = f'{season.value}_{sensor.value}'
        path = os.path.join(self.base_dir, season_dirname)
        if not os.path.exists(path):
            raise NameError(f'Could not find season {season_dirname} in base directory {self.base_dir}')
        scene_list = [os.path.basename(s) for s in glob(os.path.join(path, "*"))]
        scene_list = [int(s.split("_")[1]) for s in scene_list]
        return scene_list

    def get_patch_ids(self, season: Season, scene_id: str) -> List[int]:
        """ Returns a list of patch ids for a specific scene within a specific season
        """
        sensor = Sensor('s1')
        season_dirname = f'{season.value}_{sensor.value}'
        path = os.path.join(self.base_dir, season_dirname, f"s1_{scene_id}")
        if not os.path.exists(path):
            raise NameError(f'Could not find scene {scene_id} within season {season_dirname}')
        patch_ids = [os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(path, "*"))]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]
        return patch_ids

    def get_season_ids(self, season: Season):
        """ Return a dict of scene ids and their corresponding patch ids.
        key => scene_ids, value => list of patch_ids
        """
        ids = {}
        scene_ids = self.get_scene_ids(season)
        logging.debug(f'secne_ids:{scene_ids}')
        for sid in scene_ids:
            ids[sid] = self.get_patch_ids(season, sid)
        return ids

    def get_patch(self, season: Season, sensor: Sensor, scene_id: int, patch_id: int,
                  bands) -> Tuple[array, BoundingBox]:
        """ Returns raster data and image bounds for the defined bands of a specific patch
        This method only loads a sinlge patch from a single sensor as defined by the bands specified
        """

        if isinstance(bands, (list, tuple)):
            b = bands[0]
        else:
            b = bands

        if isinstance(b, S1Bands):
            bandEnum = S1Bands
        elif isinstance(b, S2Bands):
            bandEnum = S2Bands
        else:
            raise Exception("Invalid bands specified")

        if isinstance(bands, (list, tuple)):
            bands = [b.value for b in bands]
        else:
            bands = bands.value

        scene = f'{sensor.value}_{scene_id}'
        filename = f'{season.value}_{scene}_p{patch_id}.tif'
        patch_path = os.path.join(self.base_dir, f'{season.value}_{sensor.value}', scene, filename)

        with rasterio.open(patch_path) as patch:
            data = patch.read(bands)
            bounds = patch.bounds

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        return data, bounds

    def get_s1s2s2cloudy_triplet(self,
                                 season: Season,
                                 scene_id,
                                 patch_id,
                                 s1_bands=S1Bands.ALL,
                                 s2_bands=S2Bands.ALL,
                                 s2cloudy_bands=S2Bands.ALL):
        """ Returns a triplet of patches. S1, S2 and cloudy S2 as well as the geo-bounds of the patch
        """
        s1, bounds = self.get_patch(season, Sensor('s1'), scene_id, patch_id, s1_bands)
        s2, _ = self.get_patch(season, Sensor('s2'), scene_id, patch_id, s2_bands)
        s2cloudy, _ = self.get_patch(season, Sensor('s2_cloudy'), scene_id, patch_id, s2cloudy_bands)

        return s1, s2, s2cloudy, bounds

    def get_triplets(self,
                     season: Season,
                     scene_ids=None,
                     patch_ids=None,
                     s1_bands=S1Bands.ALL,
                     s2_bands=S2Bands.ALL,
                     s2cloudy_bands=S2Bands.ALL):
        """ Returns a triplet of numpy arrays with dimensions D, B, W, H where D is the number of patches specified
        using scene_ids and patch_ids and B is the number of bands for S1, S2 or cloudy S2
        """
        scene_list = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        s2cloudy_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, s2cloudy, bound = self.get_s1s2s2cloudy_triplet(season, sid, pid, s1_bands, s2_bands,
                                                                        s2cloudy_bands)
                s1_data.append(s1)
                s2_data.append(s2)
                s2cloudy_data.append(s2cloudy)
                bounds.append(bound)
        return np.stack(s1_data, axis=0), np.stack(s2_data, axis=0), np.stack(s2cloudy_data, axis=0), bounds

    def get_all_triplets(self) -> List[SEN12MSCRTriplet]:
        patches = []
        for season_value in Season.ALL.value:
            season = Season(season_value)
            scene_list = self.get_scene_ids(season)
            for sid in scene_list:
                patch_list = self.get_patch_ids(season, sid)
                for pid in patch_list:
                    patch = SEN12MSCRTriplet(self.base_dir, season, sid, pid, self.file_extension)
                    patches.append(patch)
        return patches

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        triplet = self.triplets[index]
        s1, s2, s2_cloudy = triplet.data
        s1, s2, s2_cloudy = np.float32(s1), np.float32(s2), np.float32(s2_cloudy)
        return np.concatenate((s1, s2_cloudy), axis=0), s2


if __name__ == "__main__":
    logging.basicConfig(filename='logger.log',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s')
    base_dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data_v2/SEN12MS_CR/'
    # Load the dataset specifying the base directory
    sen12mscr = SEN12MSCRDataset(base_dir)

    summer_ids = sen12mscr.get_season_ids(Season.SUMMER)
    cnt_patches = sum([len(pids) for pids in summer_ids.values()])
    logging.debug(f'Summer: {len(summer_ids)} seenes with a total of {cnt_patches} patches')
    print("Summer: {} scenes with a total of {} patches".format(len(summer_ids), cnt_patches))

    start = time.time()
    # Load the RGB bands of the first S2 patch in scene 8
    SCENE_ID = 80
    s2_rgb_patch, bounds = sen12mscr.get_patch(Season.SUMMER,
                                               Sensor('s2'),
                                               SCENE_ID,
                                               summer_ids[SCENE_ID][0],
                                               bands=S2Bands.RGB)
    print("Time Taken {}s".format(time.time() - start))
    print("S2 RGB: {} Bounds: {}".format(s2_rgb_patch.shape, bounds))

    print("\n")

    # Load a triplet of patches from the first three scenes of Spring - all S1 bands, NDVI S2 bands, and NDVI S2 cloudy bands
    i = 0
    start = time.time()
    for scene_id, patch_ids in summer_ids.items():
        if i >= 3:
            break

        s1, s2, s2cloudy, bounds = sen12mscr.get_s1s2s2cloudy_triplet(
            Season.SUMMER,
            scene_id,
            patch_ids[0],
            s1_bands=S1Bands.ALL,
            #   s2_bands=[S2Bands.red, S2Bands.nir1],
            s2_bands=S2Bands.ALL,
            s2cloudy_bands=[S2Bands.red, S2Bands.nir1])
        print(f"Scene: {scene_id}, S1: {s1.shape}, S2: {s2.shape}, cloudy S2: {s2cloudy.shape}, Bounds: {bounds}")
        i += 1

    print("Time Taken {}s".format(time.time() - start))

    # start = time.time()
    # print('Load all bands of all patches in a specified scene (scene 106)')
    # # Load all bands of all patches in a specified scene (scene 106)
    # s1, s2, s2cloudy, _ = sen12mscr.get_triplets(Season.SUMMER,
    #                                              100,
    #                                              s1_bands=S1Bands.ALL,
    #                                              s2_bands=S2Bands.ALL,
    #                                              s2cloudy_bands=S2Bands.ALL)

    # print(f"Scene: 106, S1: {s1.shape}, S2: {s2.shape}, cloudy S2: {s2cloudy.shape}")
    # print("Time Taken {}s".format(time.time() - start))

    # Print the number of triplets
    start = time.time()
    triplets = sen12mscr.get_all_triplets()
    print(f'Triplets Total Count is:{len(triplets)}')
    print("Time Taken {}s".format(time.time() - start))
    # Print the number of scene(ROI)
    start = time.time()
    scene_cnt = 0
    for season_value in Season.ALL.value:
        season = Season(season_value)
        ids = sen12mscr.get_scene_ids(season)
        scene_cnt += len(ids)
    print(f'Scenes Total Count is:{scene_cnt}')
    print("Time Taken {}s".format(time.time() - start))
