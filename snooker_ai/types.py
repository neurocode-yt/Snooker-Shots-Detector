"""Shared domain types for the snooker shot detection pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class EditMode(str, Enum):
    """
    STRICT — 2s before cue strike → hold until all balls stop (default recommended).
    ACTION_ONLY — short pads around action.
    NATURAL — highlight-style pre/post rolls.
    FULL_SEQUENCE — longer approach + reaction.
    """

    STRICT = "strict"
    ACTION_ONLY = "action_only"
    NATURAL = "natural"
    FULL_SEQUENCE = "full_sequence"

    @classmethod
    def from_string(cls, value: str) -> "EditMode":
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "strict_mode": cls.STRICT,
            "shots_only": cls.STRICT,
            "shot_only": cls.STRICT,
            "pure": cls.STRICT,
            "action": cls.ACTION_ONLY,
            "actiononly": cls.ACTION_ONLY,
            "highlights": cls.NATURAL,
            "natural_highlights": cls.NATURAL,
            "full": cls.FULL_SEQUENCE,
            "full_shot_sequence": cls.FULL_SEQUENCE,
            "sequence": cls.FULL_SEQUENCE,
        }
        if key in aliases:
            return aliases[key]
        return cls(key)


class CameraViewType(str, Enum):
    MAIN_TABLE = "main_table"
    WIDE_ARENA = "wide_arena"
    PLAYER_CLOSEUP = "player_closeup"
    BALL_CLOSEUP = "ball_closeup"
    SCOREBOARD = "scoreboard"
    AUDIENCE = "audience"
    COMMENTATOR = "commentator"
    REPLAY = "replay"
    SLOW_MOTION_REPLAY = "slow_motion_replay"
    ADVERTISEMENT = "advertisement"
    OTHER = "other"


class ShotState(str, Enum):
    """Temporal shot states.

    The first seven values are the canonical strict-boundary contract.  The
    remaining values are retained so older analysis JSON and integrations can
    still be loaded while callers migrate to the canonical sequence.
    """

    WAITING = "WAITING"
    CUEING = "CUEING"
    STRIKE_CANDIDATE = "STRIKE_CANDIDATE"
    STRIKE_CONFIRMED = "STRIKE_CONFIRMED"
    BALLS_MOVING = "BALLS_MOVING"
    BALLS_SETTLING = "BALLS_SETTLING"
    ALL_BALLS_STOPPED = "ALL_BALLS_STOPPED"

    # Legacy Phase-1 states (serialization compatibility).
    NO_TABLE = "NO_TABLE"
    BETWEEN_SHOTS = "BETWEEN_SHOTS"
    PLAYER_APPROACHING = "PLAYER_APPROACHING"
    PLAYER_AIMING = "PLAYER_AIMING"
    FINAL_CUEING = "FINAL_CUEING"
    BALLS_SLOWING = "BALLS_SLOWING"
    BALLS_STOPPED = "BALLS_STOPPED"
    REACTION = "REACTION"
    REPLAY = "REPLAY"
    GRAPHICS = "GRAPHICS"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class JobStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    PROXY = "proxy"
    ANALYZING = "analyzing"
    DETECTING = "detecting"
    REFINING = "refining"
    SEGMENTING = "segmenting"
    READY_FOR_REVIEW = "ready_for_review"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class VideoMetadata(BaseModel):
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    display_aspect_ratio: Optional[str] = None
    fps: float = 0.0
    is_variable_frame_rate: bool = False
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    num_audio_streams: int = 0
    rotation: float = 0.0
    time_base: Optional[str] = None
    bit_rate: Optional[int] = None
    format_name: Optional[str] = None
    has_audio: bool = False
    probe_raw: dict[str, Any] = Field(default_factory=dict)


class SceneSegment(BaseModel):
    start: float
    end: float
    view_type: CameraViewType = CameraViewType.OTHER
    cut_confidence: float = 0.0
    table_ratio: float = 0.0
    is_replay_candidate: bool = False


class FrameFeatures(BaseModel):
    """Per-sample multimodal features at a single analysis timestamp."""

    t: float
    table_confidence: float = 0.0
    table_mask_area_ratio: float = 0.0
    residual_motion_mean: float = 0.0
    residual_motion_max: float = 0.0
    motion_area_ratio: float = 0.0
    camera_motion_magnitude: float = 0.0
    scene_cut_score: float = 0.0
    view_type: CameraViewType = CameraViewType.OTHER
    green_ratio: float = 0.0
    edge_density: float = 0.0
    audio_onset: float = 0.0
    audio_rms: float = 0.0
    audio_highband: float = 0.0
    ball_count: int = 0
    cue_ball_detected: bool = False
    max_ball_speed: float = 0.0  # px/s on proxy from tracker

    # Contract-facing, ball-specific observations.  Defaults deliberately keep
    # older feature JSON loadable; producers can populate these incrementally.
    table_observable: bool = True
    observation_valid: bool = True
    ball_diameter_px: float = 0.0
    cue_ball_x: Optional[float] = None
    cue_ball_y: Optional[float] = None
    cue_ball_speed: float = 0.0
    cue_ball_normalized_speed: float = 0.0
    cue_ball_acceleration: float = 0.0
    cue_ball_track_confidence: float = 0.0
    # Optional cue geometry evidence.  When the cue is occluded these remain
    # unset and the detector falls back to stationary-ball acceleration with a
    # reduced confidence/manual-review flag.
    cue_tip_visible: bool = False
    cue_tip_distance_to_ball: float = 0.0
    cue_approach_speed: float = 0.0
    cue_forward_motion: float = 0.0
    cue_contact_score: float = 0.0
    max_ball_normalized_speed: float = 0.0
    moving_ball_count: int = 0
    occluded_ball_count: int = 0
    ball_residual_motion: float = 0.0

    state: ShotState = ShotState.UNKNOWN
    strike_score: float = 0.0
    motion_score: float = 0.0  # EMA-smoothed (strike onset)
    motion_raw: float = 0.0  # unsmoothed residual activity (ball-stop)


class StrikeCandidate(BaseModel):
    timestamp: float
    confidence: float
    evidence: dict[str, float] = Field(default_factory=dict)
    uncertainty_start: float = 0.0
    uncertainty_end: float = 0.0
    camera_view: CameraViewType = CameraViewType.OTHER
    possible_replay: bool = False


class ShotRecord(BaseModel):
    shot_id: int
    preparation_start: float = 0.0
    cue_strike: float = 0.0
    cue_strike_timestamp: float = 0.0
    ball_motion_start: float = 0.0
    ball_motion_end: float = 0.0
    clip_start: float = 0.0
    clip_end: float = 0.0
    clip_start_timestamp: float = 0.0
    clip_end_timestamp: float = 0.0
    shot_confidence: float = 0.0
    start_confidence: float = 0.0
    end_confidence: float = 0.0

    # Exact-boundary contract fields.  Legacy aliases above remain part of the
    # public model and are mirrored on input by ``_populate_compatibility_fields``.
    last_ball_motion_timestamp: float = 0.0
    physical_stop_timestamp: float = 0.0
    stop_confirmation_timestamp: float = 0.0
    strike_confidence: float = 0.0
    stop_confidence: float = 0.0
    camera_views: list[str] = Field(default_factory=list)
    possible_replay: bool = False
    manual_review_required: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.0
    included: bool = True
    user_modified: bool = False
    linked_live_shot_id: Optional[int] = None
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM

    @model_validator(mode="before")
    @classmethod
    def _populate_compatibility_fields(cls, value: Any) -> Any:
        """Accept either legacy or contract field names and expose both.

        Existing builders only know ``ball_motion_end``, ``shot_confidence``
        and ``end_confidence``.  Mirroring them here makes new contract fields
        immediately useful without breaking stored Phase-1 analyses.  When a
        caller supplies both forms explicitly, its values are preserved.
        """

        if not isinstance(value, dict):
            return value
        data = dict(value)

        legacy_end = data.get("ball_motion_end")
        contract_end = data.get("physical_stop_timestamp")
        last_motion = data.get("last_ball_motion_timestamp")
        if legacy_end is None:
            if contract_end is not None:
                data["ball_motion_end"] = contract_end
            elif last_motion is not None:
                data["ball_motion_end"] = last_motion
        elif contract_end is None:
            data["physical_stop_timestamp"] = legacy_end

        if last_motion is None:
            data["last_ball_motion_timestamp"] = data.get(
                "physical_stop_timestamp", data.get("ball_motion_end", 0.0)
            )
        if "physical_stop_timestamp" not in data:
            data["physical_stop_timestamp"] = data.get(
                "last_ball_motion_timestamp", data.get("ball_motion_end", 0.0)
            )

        for legacy, contract in (
            ("cue_strike", "cue_strike_timestamp"),
            ("clip_start", "clip_start_timestamp"),
            ("clip_end", "clip_end_timestamp"),
        ):
            if legacy not in data and contract in data:
                data[legacy] = data[contract]
            elif contract not in data and legacy in data:
                data[contract] = data[legacy]
        if "stop_confirmation_timestamp" not in data:
            data["stop_confirmation_timestamp"] = data.get("physical_stop_timestamp", 0.0)

        for legacy, contract in (
            ("shot_confidence", "strike_confidence"),
            ("end_confidence", "stop_confidence"),
        ):
            if legacy not in data and contract in data:
                data[legacy] = data[contract]
            elif contract not in data and legacy in data:
                data[contract] = data[legacy]

        return data

    def duration(self) -> float:
        return max(0.0, self.clip_end - self.clip_start)


class TimelineEvent(BaseModel):
    event_type: str
    timestamp: float
    end: Optional[float] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    job_id: str
    source_path: str
    proxy_path: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: VideoMetadata
    scenes: list[SceneSegment] = Field(default_factory=list)
    features: list[FrameFeatures] = Field(default_factory=list)
    strike_candidates: list[StrikeCandidate] = Field(default_factory=list)
    shots: list[ShotRecord] = Field(default_factory=list)
    events: list[TimelineEvent] = Field(default_factory=list)
    mode: EditMode = EditMode.STRICT
    original_duration: float = 0.0
    edited_duration: float = 0.0
    pause_removed_seconds: float = 0.0
    analysis_version: str = "0.1.1-phase1-strict"


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0  # 0..1
    stage: str = ""
    message: str = ""
    error: Optional[str] = None
    shots_detected: int = 0
    updated_at: float = 0.0


class ExportRequest(BaseModel):
    mode: EditMode = EditMode.STRICT
    output_path: Optional[str] = None
    export_clips: bool = True
    export_joined: bool = True
    export_edl: bool = True
    export_csv: bool = True
    include_replays: bool = False
    only_included: bool = True
    accurate: bool = True
    min_confidence: float = 0.0
    min_importance: float = 0.0


class ShotUpdate(BaseModel):
    clip_start: Optional[float] = None
    clip_end: Optional[float] = None
    cue_strike: Optional[float] = None
    ball_motion_end: Optional[float] = None
    included: Optional[bool] = None
    possible_replay: Optional[bool] = None
    manual_review_required: Optional[bool] = None
    preparation_start: Optional[float] = None


class NewShot(BaseModel):
    cue_strike: float
    clip_start: float
    clip_end: float
    ball_motion_end: Optional[float] = None
    preparation_start: Optional[float] = None
    shot_confidence: float = 1.0
