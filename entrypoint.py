from utils.results import load_data_target
import os
import sys
import django
sys.path.extend([WORKING_DIR_AND_PYTHON_PATHS])

os.environ['DJANGO_SETTINGS_MODULE'] = 'ddyxpm.settings'

django.setup()
globals().update(
    {_class.__name__: _class for _class in django.apps.apps.get_models()})
