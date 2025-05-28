from yeval.task.hendrycks_math import MATHBaseTask
from yeval.task import register_task, YevalTask

@register_task("train:math_algebra")
class TrainMATHAlgebraTask(MATHBaseTask):
    test_split='train'
    data_name='algebra'

@register_task("train:math_counting_and_probability")
class TrainMATHCountingAndProbabilityTask(TrainMATHAlgebraTask):
    data_name='counting_and_probability'

@register_task("train:math_geometry")
class TrainMATHGeometryTask(TrainMATHAlgebraTask):
    data_name='geometry'

@register_task("train:math_intermediate_algebra")
class TrainMATHIntermediateAlgebraTask(TrainMATHAlgebraTask):
    data_name='intermediate_algebra'

@register_task("train:math_number_theory")
class TrainMATHNumberTheoryTask(TrainMATHAlgebraTask):
    data_name='number_theory'

@register_task("train:math_prealgebra")
class TrainMATHPrealgebraTask(TrainMATHAlgebraTask):
    data_name='prealgebra'

@register_task("train:math_precalculus")
class TrainMATHPrecalculusTask(TrainMATHAlgebraTask):
    data_name='precalculus'
