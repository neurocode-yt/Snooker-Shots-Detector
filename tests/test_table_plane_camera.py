"""A table-plane consensus may correct a background zoom, with bounded fit."""

import cv2
import numpy as np
import pytest

from snooker_ai.motion.camera import CameraMotionEstimator


def mocked_tracking(monkeypatch,global_matrix,table_matrix,bad=False):
    rng=np.random.default_rng(8)
    background=rng.uniform([10,10],[450,250],(80,2)).astype(np.float32)
    centres=np.array([[150,120],[270,140],[200,210]],np.float32)
    table=np.vstack([c+rng.uniform(-5,5,(12,2)) for c in centres]).astype(np.float32)
    def move(points,matrix):
        return (points@matrix[:,:2].T+matrix[:,2]).astype(np.float32)
    background_next=move(background,global_matrix)
    table_next=move(table,table_matrix)
    if bad:
        table_next[:20]+=rng.uniform(-20,20,(20,2))
    calls=[0]
    def corners(*args,**kwargs):
        points=background if calls[0]==0 else table
        calls[0]+=1
        return points.reshape(-1,1,2)
    def track(first,second,points,unused,**kwargs):
        target=background_next if len(points)==len(background) else table_next
        return target.reshape(-1,1,2),np.ones((len(points),1),np.uint8),None
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',corners)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',track)


@pytest.mark.parametrize('bad',[False,True])
def test_table_registration_requires_independent_inlier_consensus(config,monkeypatch,bad):
    global_matrix=np.array([[1.008,0.,-2.],[0.,1.008,-1.5]])
    table_matrix=np.array([[1.014,.004,-4.],[.002,1.022,-3.]])
    mocked_tracking(monkeypatch,global_matrix,table_matrix,bad)
    estimator=CameraMotionEstimator(config)
    estimator.use_opencl=False
    image=np.zeros((600,960),np.uint8)
    result=estimator.estimate(image,image,static_regions=[(300,240,30),(540,280,30),(400,420,30)])
    assert not result.is_cut_like
    expected=(global_matrix if bad else table_matrix).copy()
    expected[:,2]/=estimator.estimation_scale
    assert result.transform==pytest.approx(expected,abs=.01)
