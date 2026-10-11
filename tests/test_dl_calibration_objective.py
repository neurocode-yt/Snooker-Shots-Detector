import numpy as np
import pytest


def test_balanced_calibration_does_not_buy_zero_misses_with_many_extra_events():
    from snooker_ai.dl.training import TrainingConfig, _PreparedVideo, _calibration
    from snooker_ai.dl.dataset import build_targets
    times=np.arange(0.,32.,.125)
    entry={'id':'validation','group_id':'validation','duration':32.,
           'reviewed_intervals':[{'start':0.,'end':32.}],
           'contacts':[{'lower':5.,'upper':5.125,'provenance':'source'},
                       {'lower':29.,'upper':29.125,'provenance':'source'}], 'clips':[],'labels':[]}
    targets=build_targets(times,entry)
    prepared=_PreparedVideo(entry,times,np.zeros((len(times),4)),targets.targets,targets.mask,
                            targets.contact_bags,targets.contact_weights)
    probabilities=np.zeros((len(times),5),dtype=np.float32)
    probabilities[int(5/.125),0]=.9
    probabilities[int(29/.125),0]=.09
    for timestamp in [1.,3.,7.,9.,11.,13.,15.,17.,19.,21.,23.,25.]:
        probabilities[int(timestamp/.125),0]=.2
    zero_miss=_calibration([prepared],[probabilities],TrainingConfig(target_event_recall=1.))['event']
    balanced=_calibration([prepared],[probabilities],TrainingConfig(target_event_recall=1.,calibration_objective='f2'))['event']
    assert zero_miss['true_positive']==2 and zero_miss['false_positive']==12
    assert balanced['true_positive']==1 and balanced['false_positive']==0
    assert balanced['f2']>zero_miss['f2']


@pytest.mark.parametrize('options',[{'calibration_objective':'unknown'},{'selection_objective':'unknown'}])
def test_unknown_model_objectives_cannot_silently_change_training(options):
    from snooker_ai.dl.training import TrainingConfig
    with pytest.raises(ValueError,match='objective'):
        TrainingConfig(**options)
