from django.db import models

# Create your models here.
class LottoDraw(models.Model):
    class Meta:
        verbose_name = "Lotto Draw"
        verbose_name_plural = "Lotto Draws"

    name = models.CharField(max_length=64, verbose_name="Bias Name")
    bias_weight = models.FloatField(default=0.0)
    ball1 = models.IntegerField(default=1) 
    ball2 = models.IntegerField(default=1)
    ball3 = models.IntegerField(default=1)
    ball4 = models.IntegerField(default=1)
    ball5 = models.IntegerField(default=1)
    ball6 = models.IntegerField(default=1)

    ball = models.IntegerField(default=1)
    draw_number = models.IntegerField(default=1)
    draw_date = models.DateTimeField()



