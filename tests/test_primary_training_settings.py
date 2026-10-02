"""Portable settings and config compatibility without installing GPU libraries."""

from unittest.mock import patch

import gradio as gr
import pytest

from kohya_gui.class_basic_training import BasicTraining
from kohya_gui import common_gui, dreambooth_gui, lora_gui
from kohya_gui.class_gui_config import KohyaSSGUIConfig


def build_basic(config):
    with gr.Blocks() as app:
        basic = BasicTraining(sdxl_checkbox=gr.Checkbox(False), config=config)
    return app, basic


def test_batch_config_sets_value_not_slider_increment():
    _, basic = build_basic({"basic.train_batch_size": 4})
    assert basic.train_batch_size.value == 4
    assert basic.train_batch_size.step == 1


@pytest.mark.parametrize(
    "optimizer",
    ["AdamW8bit", "Lion8bit", "PagedAdamW32bit", "bitsandbytes.optim.AdEMAMix8bit"],
)
def test_missing_dependency_detects_aliases_and_qualified_names(optimizer):
    with patch("importlib.util.find_spec", return_value=None):
        assert common_gui.optimizer_dependency_error(optimizer)
        assert not common_gui.optimizer_dependency_error("AdamW")
        assert not common_gui.optimizer_dependency_error("my_optimizer.Custom")


def test_missing_dependency_uses_available_default_and_preserves_loaded_config():
    with patch("importlib.util.find_spec", return_value=None):
        _, fresh = build_basic({})
        _, loaded = build_basic({"basic.optimizer": "AdamW8bit"})
    assert fresh.optimizer.value == "AdamW"
    assert "AdamW8bit" not in [value for _, value in fresh.optimizer.choices]
    assert not fresh.optimizer_status.visible
    assert loaded.optimizer.value == "AdamW8bit"
    assert loaded.optimizer_status.visible
    assert "bitsandbytes" in loaded.optimizer_status.value


def test_installed_dependency_keeps_original_default():
    with patch("importlib.util.find_spec", return_value=object()):
        _, basic = build_basic({})
    assert basic.optimizer.value == "AdamW8bit"
    assert "AdamW8bit" in [value for _, value in basic.optimizer.choices]


@pytest.mark.parametrize(
    "epochs,steps,mode",
    [
        (0, 1600, "Steps"),
        (5, 0, "Epochs"),
        (5, 1600, "Epochs"),
        (0, 0, "Automatic (legacy)"),
    ],
)
def test_old_configs_keep_values_and_resolve_existing_precedence(epochs, steps, mode):
    _, basic = build_basic(
        {
            "basic.epoch": 3,
            "basic.max_train_epochs": epochs,
            "basic.max_train_steps": steps,
        }
    )
    assert basic.duration_mode.value == mode
    assert basic.epoch.value == 3
    assert basic.max_train_epochs.value == epochs
    assert basic.max_train_steps.value == steps
    assert basic.epoch.visible == (mode == "Automatic (legacy)")
    assert basic.max_train_epochs.visible == (mode == "Epochs")
    assert basic.max_train_steps.visible == (mode == "Steps")


def test_user_mode_switches_clear_the_other_limit():
    assert common_gui.select_training_duration("Epochs", 3, 0, 1000) == (3, 3, 0)
    assert common_gui.select_training_duration("Steps", 3, 5, 1000) == (3, 0, 1000)
    assert common_gui.select_training_duration("Steps", 3, 5, 0) == (3, 0, 1600)
    assert common_gui.select_training_duration("Automatic (legacy)", 3, 5, 1000) == (
        3,
        0,
        0,
    )


@pytest.mark.parametrize(
    "module,builder", [(dreambooth_gui, "dreambooth_tab"), (lora_gui, "lora_tab")]
)
def test_real_gui_entry_blocks_unavailable_optimizer_before_training(module, builder):
    with gr.Blocks():
        getattr(module, builder)(headless=True, config=KohyaSSGUIConfig())
    entries = module.last_built_gui_entries
    data = {
        component: component.value for _, component in module.last_built_field_registry
    }
    components = entries["components"]
    data[components["dummy_headless"]] = True
    data[components["dummy_db_false"]] = False
    data[components["dummy_db_true"]] = True
    optimizer = dict(module.last_built_field_registry)["optimizer"]
    data[optimizer] = "AdamW8bit"
    fields = dict(module.last_built_field_registry)
    data[fields["output_dir"]] = "output"
    data[fields["pretrained_model_name_or_path"]] = "model"
    data[fields["train_data_dir"]] = "images"
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch.object(module, "train_model") as train,
    ):
        with pytest.raises(gr.Error, match="bitsandbytes"):
            entries["train_model"](data)
        train.assert_not_called()
        entries["print_command"](data)
        train.assert_called_once()
        assert train.call_args.kwargs["optimizer"] == "AdamW8bit"


def test_missing_required_fields_use_in_page_errors():
    settings = {
        "optimizer": "AdamW",
        "output_dir": "",
        "pretrained_model_name_or_path": "model",
        "train_data_dir": "images",
    }
    with pytest.raises(gr.Error, match="output directory"):
        common_gui.validate_primary_training_settings(settings)


def test_mode_round_trip_remembers_user_limits():
    epoch, epochs, steps, remembered_epochs, remembered_steps = (
        common_gui.switch_training_duration("Epochs", 2, 0, 2300, 7, 1600)
    )
    assert (epoch, epochs, steps) == (7, 7, 0)
    assert remembered_steps == 2300
    result = common_gui.switch_training_duration(
        "Steps", epoch, epochs, steps, remembered_epochs, remembered_steps
    )
    assert result[:3] == (7, 0, 2300)


def test_preview_checks_missing_paths_without_requiring_local_optimizer():
    with pytest.raises(gr.Error, match="output directory"):
        common_gui.validate_primary_training_settings(
            {"optimizer": "AdamW8bit", "output_dir": ""}, check_optimizer=False
        )
