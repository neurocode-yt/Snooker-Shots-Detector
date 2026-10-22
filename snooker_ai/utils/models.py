"""Shared model API, including the pure-Python runtime used on Android.

Desktop installations use Pydantic 2 unchanged. Android currently has no
Pydantic-core wheel, so it uses Pydantic 1's pure-Python validators behind this
small compatibility layer. Keep serialization and validation in one place.
"""

from __future__ import annotations

import json

import pydantic

__all__ = ["BaseModel", "ConfigDict", "Field", "field_validator", "model_validator"]

if hasattr(pydantic, "model_validator"):
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
else:
    from pydantic import Field, root_validator, validator
    from pydantic.main import ModelMetaclass

    ConfigDict = dict

    class _ModelMeta(ModelMetaclass):
        def __new__(mcs, name, bases, namespace, **kwargs):
            # Pydantic 2 calls collection-size constraints min/max_length;
            # Pydantic 1 calls them min/max_items.
            from pydantic.fields import FieldInfo
            for field, annotation in namespace.get("__annotations__", {}).items():
                info = namespace.get(field)
                if isinstance(info, FieldInfo) and str(annotation).startswith(("list[", "set[", "tuple[")):
                    for bound in ("min", "max"):
                        length = getattr(info, bound + "_length")
                        if length is not None:
                            setattr(info, bound + "_items", length)
                            setattr(info, bound + "_length", None)
            config = namespace.pop("model_config", None)
            if config is not None:
                namespace["Config"] = type("Config", (), config)
            return super().__new__(mcs, name, bases, namespace, **kwargs)

    class BaseModel(pydantic.BaseModel, metaclass=_ModelMeta):
        @classmethod
        def model_validate(cls, value, **kwargs):
            return cls.parse_obj(value)

        @classmethod
        def model_validate_json(cls, value, **kwargs):
            return cls.parse_raw(value)

        def model_dump(self, *, mode="python", **kwargs):
            return json.loads(self.json(**kwargs)) if mode == "json" else self.dict(**kwargs)

        def model_dump_json(self, **kwargs):
            return self.json(**kwargs)

        def model_copy(self, *, update=None, deep=False):
            return self.copy(update=update, deep=deep)

        @property
        def model_fields_set(self):
            return self.__fields_set__

    def model_validator(*, mode):
        if mode != "before":
            raise ValueError("Android compatibility supports before model validators")

        def decorate(method):
            return root_validator(pre=True, allow_reuse=True)(
                method.__func__ if isinstance(method, classmethod) else method
            )
        return decorate

    def field_validator(*fields, mode="after"):
        def decorate(method):
            return validator(*fields, pre=mode == "before", allow_reuse=True)(
                method.__func__ if isinstance(method, classmethod) else method
            )
        return decorate
