from __future__ import annotations

from ui_qt.dialogs.form_schema import (
    FieldSpec,
    default_entity_fields,
    normalize_field_specs,
    parse_float,
    parse_int,
    parse_text,
    payload_from_strings,
)
from ui_qt.dialogs.record_form_dialog import RecordFormDialog
from ui_qt.dialogs.configuration_details_dialog import ConfigurationDetailsDialog

__all__ = [
    "FieldSpec",
    "ConfigurationDetailsDialog",
    "RecordFormDialog",
    "default_entity_fields",
    "normalize_field_specs",
    "parse_float",
    "parse_int",
    "parse_text",
    "payload_from_strings",
]
