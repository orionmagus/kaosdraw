from django.db import models
from lotto.managers import ResultsQuerySet
from utils.numbers import NumPool
import json
import numpy as np

# Create your models here.
class Powerball(models.Model):
    class Meta:
        verbose_name = "Powerball Draw"
        verbose_name_plural = "Powerball Draws"
    draw_number = models.IntegerField(default=1, primary_key=True)
    draw_type = models.CharField(max_length=64, default='6/50')
    draw_date = models.DateTimeField()
    ball1 = models.IntegerField(default=1)
    ball2 = models.IntegerField(default=1)
    ball3 = models.IntegerField(default=1)
    ball4 = models.IntegerField(default=1)
    ball5 = models.IntegerField(default=1)

    result = models.TextField()
    power_ball = models.IntegerField(default=1)
    # result = ResultField()

    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    objects = models.Manager.from_queryset(ResultsQuerySet)

    # def save(self, *args, **kw):
    #     v = {k: getattr(self, k) for k in ('ball1', 'ball2', 'ball3', 'ball4', 'ball5')}
    #     np = NumPool(value=v, shape=(5, 50))
    #     self.result = json.dumps(np.array(v).tolist())
    #     super(Powerball, self).save(*args, **kw)
"""
from utils.results import update_datapb, date
update_datapb(None)
update_datapb(date(2014,1,1))
"""

