"""イベント検出の設定。"""

from dataclasses import dataclass


@dataclass
class EventConfig:
    """ターンテイキングイベント検出の設定。"""

    min_context_time: float = 3
    metric_time: float = 0.2
    metric_pad_time: float = 0.05
    max_time: int = 20
    frame_hz: int = 20
    equal_hold_shift: int = 1
    prediction_region_time: float = 0.5

    # Shift/Hold
    sh_pre_cond_time: float = 2.0
    sh_post_cond_time: float = 2.0
    sh_prediction_region_on_active: bool = True

    # Backchannel
    bc_pre_cond_time: float = 1.0
    bc_post_cond_time: float = 2.0
    bc_max_duration: float = 1.0
    bc_negative_pad_left_time: float = 1.0
    bc_negative_pad_right_time: float = 2.0

    # Long/Short
    long_onset_region_time: float = 0.2
    long_onset_condition_time: float = 1.0

    @staticmethod
    def add_argparse_args(parser, fields_added=[]):
        for k, v in EventConfig.__dataclass_fields__.items():
            parser.add_argument(f"--event_{k}", type=v.type, default=v.default)
            fields_added.append(k)
        return parser, fields_added

    @staticmethod
    def args_to_conf(args):
        return EventConfig(
            **{
                k.replace("event_", ""): v
                for k, v in vars(args).items()
                if k.startswith("event_")
            }
        )
