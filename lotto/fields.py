from django.db import models
from utils.numbers import NumPool, BallInt
import json


class ResultField(models.IntegerField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, context):
        if value is None:
            return value
        # print(expression, connection, context)
        return int(ResultInt(value))

    def to_python(self, value):
        if isinstance(value, NumPool):
            return value
        if value is None:
            return value
        return NumPool(**value)

    def get_prep_value(self, value):
        return NumPool(value)


class NeuronWeightField(models.IntegerField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection, context):
        if value is None:
            return value
        # print(expression, connection, context)
        return parse_shape(value)

    def to_python(self, value):
        if is_shape(value):
            return value
        if value is None:
            return value
        return parse_shape(value)

    def get_prep_value(self, value):
        return serialize_shape(value)
