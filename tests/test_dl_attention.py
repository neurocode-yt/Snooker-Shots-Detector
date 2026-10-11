"""Behavioral contracts for optional localized neural motion and RGB changes."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from snooker_ai.dl.temporal import (  # noqa: E402
    TemporalHighlightEnsemble,
    TemporalHighlightNet,
    TemporalModelConfig,
    checkpoint_payload,
    load_temporal_checkpoint,
    predict_probabilities,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(306)


def attention_model(*, lag=0, grid=(5, 7), channels=8):
    return TemporalHighlightNet(TemporalModelConfig(
        input_dim=4 + channels*grid[0]*grid[1], hidden_dim=8,
        flow_grid=grid, flow_channels=channels, dropout=0,
        motion_pooling='rgb_attention', rgb_difference_frames=lag,
    )).eval()


@pytest.mark.parametrize('lag', [1, 3, 32])
def test_rgb_change_history_matches_full_spans_across_chunks_and_gaps(lag):
    model=attention_model(lag=lag)
    rng=np.random.default_rng(7)
    lengths=[185,17,151]
    values=rng.normal(size=(sum(lengths),model.config.input_dim)).astype(np.float32)
    # Large source discontinuities make leaking previous RGB history visible.
    values[185:202,:4]+=25
    values[202:,:4]-=25
    times=np.concatenate([np.arange(185)/8,100+np.arange(17)/8,400+np.arange(151)/8])
    expected=[]
    position=0
    with torch.inference_mode():
        for length in lengths:
            expected.append(torch.sigmoid(model(torch.from_numpy(values[position:position+length])[None])[0]).numpy())
            position+=length
    expected=np.concatenate(expected)
    for chunk in [1,11,43,512]:
        actual=predict_probabilities(model,values,timestamps=times,chunk_frames=chunk)
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)


def capture_motion_input(model,features):
    observed=[]
    handle=model.motion_projection.register_forward_pre_hook(
        lambda _module,args: observed.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            model(features)
    finally:
        handle.remove()
    return observed[0]


def position_sensitive_fixture():
    model=attention_model(grid=(7,11))
    # An identity local filter isolates whether pooling retains spatial location.
    with torch.no_grad():
        for layer in [model.motion_encoder[0],model.motion_encoder[2]]:
            layer.weight.zero_()
            layer.bias.zero_()
            layer.weight[0,0,1,1]=1
        model.motion_key.weight.zero_()
        model.motion_key.bias.zero_()
        model.motion_key.weight[0,-2]=1  # Learned horizontal position preference.
        model.motion_query.weight.zero_()
        model.motion_query.bias.zero_()
        model.motion_query.bias[::32]=12
    return model


def test_attention_distinguishes_equal_motion_at_different_spatial_positions():
    model=position_sensitive_fixture()
    left=torch.zeros(1,3,model.config.input_dim)
    right=left.clone()
    left[...,4:].reshape(1,3,8,7,11)[:,:,0,3,3]=8
    right[...,4:].reshape(1,3,8,7,11)[:,:,0,3,7]=8
    pooled_left=capture_motion_input(model,left)
    pooled_right=capture_motion_input(model,right)
    # The global summaries are identical; localized attention must keep the distinction.
    torch.testing.assert_close(pooled_left[:,:64],pooled_right[:,:64],atol=0,rtol=0)
    assert not torch.allclose(pooled_left[:,64:],pooled_right[:,64:])


def test_rgb_query_changes_motion_localization_with_motion_held_fixed():
    model=position_sensitive_fixture()
    with torch.no_grad():
        model.rgb_projection[0].weight.zero_()
        model.rgb_projection[0].bias.zero_()
        model.rgb_projection[0].weight[0,0]=1
        model.motion_query.bias.zero_()
        model.motion_query.weight[::32,0]=8
    first=torch.zeros(1,2,model.config.input_dim)
    first[...,0]=1
    first[...,4:].reshape(1,2,8,7,11)[:,:,0,3,7]=8
    second=first.clone()
    second[...,0]=-1
    a,b=capture_motion_input(model,first),capture_motion_input(model,second)
    torch.testing.assert_close(a[:,:64],b[:,:64],atol=0,rtol=0)
    assert not torch.allclose(a[:,64:],b[:,64:])


def test_rgb_difference_respects_frozen_affine_normalization():
    original=attention_model(lag=4)
    transformed=attention_model(lag=4)
    transformed.load_state_dict(original.state_dict())
    rng=np.random.default_rng(32)
    dim=original.config.input_dim
    mean=rng.normal(size=dim).astype(np.float32)
    scale=rng.uniform(.2,3,size=dim).astype(np.float32)
    gain=rng.uniform(.5,4,size=dim).astype(np.float32)
    offset=rng.normal(size=dim).astype(np.float32)
    original.set_feature_normalization(mean,scale)
    transformed.set_feature_normalization(gain*mean+offset,gain*scale)
    values=rng.normal(size=(130,dim)).astype(np.float32)
    a=predict_probabilities(original,values,chunk_frames=13)
    b=predict_probabilities(transformed,values*gain+offset,chunk_frames=29)
    np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)


def test_eight_flow_channels_receive_gradients_and_checkpoint_roundtrip(tmp_path):
    model=attention_model(lag=3)
    features=torch.randn(2,35,model.config.input_dim,requires_grad=True)
    model(features).square().mean().backward()
    motion_gradient=features.grad[...,4:].reshape(2,35,8,5,7)
    assert torch.isfinite(motion_gradient).all()
    assert torch.count_nonzero(motion_gradient[:,:,6:])>0
    assert model.rgb_projection[0].in_features==8
    assert model.motion_encoder[0].in_channels==8
    spec={'version':2,'dimension':model.config.input_dim,'flow_grid':[5,7],
          'flow_channel_count':8,'sample_fps':8}
    path=tmp_path/'attention.pt'
    torch.save(checkpoint_payload(model,feature_spec=spec,dataset_fingerprint='fixture',
        training={'split_groups':{'train':['fixture']}},calibration={}),path)
    # The public loader must retain tensor-only safe deserialization and every option.
    safe_payload=torch.load(path,weights_only=True,map_location='cpu')
    assert safe_payload['model_config']==asdict(model.config)
    restored,metadata=load_temporal_checkpoint(path,expected_feature_spec=spec)
    assert restored.config==model.config and metadata['feature_spec']==spec
    values=features.detach().numpy()[0]
    np.testing.assert_allclose(predict_probabilities(restored,values,chunk_frames=7),
                               predict_probabilities(model,values,chunk_frames=64),atol=2e-6,rtol=2e-6)


def test_mixed_legacy_and_attention_ensemble_preserves_each_members_context(tmp_path):
    localized=attention_model(lag=16)
    legacy=TemporalHighlightNet(TemporalModelConfig(
        input_dim=localized.config.input_dim,hidden_dim=8,flow_grid=(5,7),flow_channels=8,dropout=0,
    )).eval()
    ensemble=TemporalHighlightEnsemble([legacy,localized],[.25,.75],aggregation='max_event').eval()
    values=np.random.default_rng(3).normal(size=(300,ensemble.config.input_dim)).astype(np.float32)
    a=predict_probabilities(legacy,values,chunk_frames=512)
    b=predict_probabilities(localized,values,chunk_frames=512)
    expected=.25*a+.75*b
    expected[:,0]=np.maximum(a[:,0],b[:,0])
    np.testing.assert_allclose(predict_probabilities(ensemble,values,chunk_frames=17),expected,atol=2e-6,rtol=2e-6)
    path=tmp_path/'mixed.pt'
    torch.save(checkpoint_payload(ensemble,feature_spec={'dimension':ensemble.config.input_dim},
        dataset_fingerprint='fixture',training={},calibration={}),path)
    restored,_=load_temporal_checkpoint(path)
    np.testing.assert_allclose(predict_probabilities(restored,values,chunk_frames=37),expected,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('options',[
    {'motion_pooling':'unknown'},{'motion_pooling':'rgb_attention','use_rgb':False},
    {'rgb_difference_frames':True},{'rgb_difference_frames':-1},
    {'rgb_difference_frames':33},{'rgb_difference_frames':1.5},
])
def test_invalid_attention_or_difference_options_fail_before_model_creation(options):
    from snooker_ai.dl.training import TrainingConfig
    with pytest.raises(ValueError):
        TemporalModelConfig(input_dim=284,flow_grid=(5,7),flow_channels=8,**options)
    with pytest.raises(ValueError):
        TrainingConfig(**options)
