# Generated by Django 3.0.3 on 2020-02-16 01:21

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='LottoDraw',
            fields=[
                ('draw_number', models.IntegerField(default=1, primary_key=True, serialize=False)),
                ('lotto_type', models.CharField(default='6/52', max_length=64)),
                ('draw_date', models.DateTimeField()),
                ('ball1', models.IntegerField(default=1)),
                ('ball2', models.IntegerField(default=1)),
                ('ball3', models.IntegerField(default=1)),
                ('ball4', models.IntegerField(default=1)),
                ('ball5', models.IntegerField(default=1)),
                ('ball6', models.IntegerField(default=1)),
                ('bonus_ball', models.IntegerField(default=1)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Lotto Draw',
                'verbose_name_plural': 'Lotto Draws',
            },
        ),
        migrations.CreateModel(
            name='NeuralNetGroup',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=64, verbose_name='Net Group Name')),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
                ('accuracy', models.FloatField(default=0.0)),
            ],
            options={
                'verbose_name': 'NeuralNet Group',
                'verbose_name_plural': 'NeuralNet Group',
            },
        ),
        migrations.CreateModel(
            name='NeuralModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('net_ident', models.CharField(max_length=64, verbose_name='Net ID')),
                ('tech_class', models.CharField(default='utils.neural_net.NeuralNetwork', max_length=255, verbose_name='Net Model')),
                ('position', models.IntegerField(default=1)),
                ('inputs', models.IntegerField(default=1)),
                ('shape', models.TextField(verbose_name='Layers IO')),
                ('loss', models.TextField(verbose_name='loss Function')),
                ('metrics', models.TextField(verbose_name='Metric Function')),
                ('learning_rate', models.FloatField(default=0.0)),
                ('bias', models.FloatField(default=None, null=True)),
                ('accuracy', models.FloatField(default=0.0)),
                ('training_epochs', models.IntegerField(default=1)),
                ('node_weights', models.TextField(null=True, verbose_name='Neuron Weights')),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('modified', models.DateTimeField(auto_now=True)),
                ('group', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='nets', to='lotto.NeuralNetGroup')),
            ],
            options={
                'verbose_name': 'NeuralNet Model',
                'verbose_name_plural': 'NeuralNet Models',
            },
        ),
    ]
