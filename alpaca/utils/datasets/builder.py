from .boston_housing import BostonHousingData
from .concrete import ConcreteData
from .energy_efficiency import EnergyEfficiencyData
from .kin8nm import Kin8nmData
from .naval_propulsion import NavalPropulsionData
from .ccpp import CCPPData
from .protein_structure import ProteinStructureData
from .red_wine import RedWineData
from .yacht_hydrodynamics import YachtHydrodynamicsData
from .year_prediction_msd import YearPredictionMSDData
from .openml import MnistData, FashionMnistData, Cifar10, SVHN
from .fb_comment import FBCommentData
from .superconduct import SuperconductData
from .bike_sharing import BikeSharingHourData
from .ailerons import AileronsData
from .sgemm_gpu import SGEMMGPUData
from .ct_slices import CTSlicesData

DATASETS = {
    "boston_housing": BostonHousingData,
    "concrete": ConcreteData,
    "energy_efficiency": EnergyEfficiencyData,
    "kin8nm": Kin8nmData,
    "naval_propulsion": NavalPropulsionData,
    "ccpp": CCPPData,
    "protein_structure": ProteinStructureData,
    "red_wine": RedWineData,
    "yacht_hydrodynamics": YachtHydrodynamicsData,
    "year_prediction_msd": YearPredictionMSDData,
    "mnist": MnistData,
    "fashion_mnist": FashionMnistData,
    "cifar_10": Cifar10,
    "svhn": SVHN,
    "fb_comment": FBCommentData,
    "superconduct": SuperconductData,
    "bike_sharing": BikeSharingHourData,
    "ailerons": AileronsData,
    "sgemm_gpu": SGEMMGPUData,
    "ct_slices": CTSlicesData
}


def build_dataset(name, **kwargs):
    return DATASETS[name](**kwargs)
