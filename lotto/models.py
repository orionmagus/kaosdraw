from django.db import models
from lotto.managers import ResultsQuerySet
# from lotto.fields import NeuronWeightField
# Create your models here.


class LottoDraw(models.Model):
    class Meta:
        verbose_name = "Lotto Draw"
        verbose_name_plural = "Lotto Draws"
    draw_number = models.IntegerField(default=1, primary_key=True)
    lotto_type = models.CharField(max_length=64, default='6/52')
    draw_date = models.DateTimeField()
    ball1 = models.IntegerField(default=1)
    ball2 = models.IntegerField(default=1)
    ball3 = models.IntegerField(default=1)
    ball4 = models.IntegerField(default=1)
    ball5 = models.IntegerField(default=1)
    ball6 = models.IntegerField(default=1)

    bonus_ball = models.IntegerField(default=1)
    # result = ResultField()

    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    objects = models.Manager.from_queryset(ResultsQuerySet)


class NeuralNetGroup(models.Model):
    class Meta:
        verbose_name = "NeuralNet Group"
        verbose_name_plural = "NeuralNet Group"

    name = models.CharField(max_length=64, verbose_name="Net Group Name")
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    accuracy = models.FloatField(default=0.0)


class NeuralModel(models.Model):
    class Meta:
        verbose_name = "NeuralNet Model"
        verbose_name_plural = "NeuralNet Models"
    group = models.ForeignKey(
        NeuralNetGroup, on_delete=models.CASCADE, related_name='nets')
    net_ident = models.CharField(max_length=64, verbose_name="Net ID")
    tech_class = models.CharField(
        default='utils.neural_net.NeuralNetwork', max_length=255, verbose_name="Net Model")

    position = models.IntegerField(default=1)
    inputs = models.IntegerField(default=1)
    shape = models.TextField(verbose_name="Layers IO")
    loss = models.TextField(verbose_name="loss Function")
    metrics = models.TextField(verbose_name="Metric Function")
    learning_rate = models.FloatField(default=0.0)
    bias = models.FloatField(default=None, null=True)
    accuracy = models.FloatField(default=0.0)
    training_epochs = models.IntegerField(default=1)
    node_weights = models.TextField(verbose_name="Neuron Weights", null=True)
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)


# class NeuralLayer(models.Model):
#     class Meta:
#         verbose_name = "NeuralLayer"
#         verbose_name_plural = "NeuralLayer"
#     model = models.ForeignKey(
#         NeuralModel, on_delete=models.CASCADE, related_name='nets')
#     tech_class = models.CharField(
#         default='utils.neural_net.NeuralNetwork', max_length=255, verbose_name="Layer Class")
#     inputs = models.IntegerField(default=1)
