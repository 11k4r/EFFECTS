from effects.preprocess.resampler import Resampler
from effects.preprocess.anomaly_detector import LowessAnomalyDetector
from effects.preprocess.preprocessor import Preprocessor
from effects.extraction.slicers import Slicer, random_slicer_generator
from effects.extraction.transforms import identity_transform, diff_transform, standard_transform, scale_transform, power_transform, fft_transform, sub_transform, div_transform
from effects.extraction.aggregations import maxAgg, minAgg, meanAgg, ptpAgg, stdAgg, sumAgg, trendAgg, varAgg, medianAgg, peakAgg

DEFAULT_RESAMPLER = Resampler()
DEFAULT_ANOMALY_DETECTOR = LowessAnomalyDetector()

SEPARATOR = '__'
PREPROCESSOR = Preprocessor(DEFAULT_RESAMPLER, DEFAULT_ANOMALY_DETECTOR)
U_TRANSFORMS = [identity_transform, diff_transform, standard_transform, scale_transform, power_transform, fft_transform]
B_TRANSFORMS = [sub_transform, div_transform]
SLICER = Slicer(random_slicer_generator(10), 'random')
AGG_FUNCTIONS = [maxAgg, minAgg, meanAgg, ptpAgg, stdAgg, sumAgg, trendAgg, varAgg, medianAgg, peakAgg]
