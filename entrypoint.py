from utils.results import load_data_target, update_data
from utils.results import load_data_target
import os
from os.path import (
    abspath,
    dirname
)
import sys
import django
sys.path.extend(dirname(abspath(__file__)))

os.environ['DJANGO_SETTINGS_MODULE'] = 'kaosdraw.settings'

django.setup()
app_models = {
    _class.__name__: _class for _class in django.apps.apps.get_models()
}
globals().update(app_models)
