from yeval.task.hendrycks_math import MATHBaseTask
from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.metrics import math_eval

@register_task("train:math_algebra")
class TrainMATHAlgebraTask(MATHBaseTask):
    preprocessing=None
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

class MATH500BaseTask(YevalTask):
    data_path="HuggingFaceH4/MATH-500"
    input_text=lambda x: x["problem"]
    output_text=lambda x: x["answer"]
    test_split="test"
    evaluation={"accuracy": math_eval}
    logging=log_token_usage

@register_task("math500_algebra")
class MATH500AlgebraTask(MATH500BaseTask):
    preprocessing=lambda dataset: dataset.filter(lambda x: x["subject"] == "Algebra")

@register_task("math500_counting_and_probability")
class MATH500CountingAndProbabilityTask(MATH500BaseTask):
    preprocessing=lambda dataset: dataset.filter(lambda x: x["subject"] == "Counting & Probability")

@register_task("math500_geometry")
class MATH500GeometryTask(MATH500BaseTask):
    preprocessing=lambda dataset: dataset.filter(lambda x: x["subject"] == "Geometry")

@register_task("math500_intermediate_algebra")
class MATH500IntermediateAlgebraTask(MATH500BaseTask):
    preprocessing=lambda dataset: dataset.filter(lambda x: x["subject"] == "Intermediate Algebra")

@register_task("math500_number_theory")
class MATH500NumberTheoryTask(MATH500BaseTask):
    preprocessing=lambda dataset: dataset.filter(lambda x: x["subject"] == "Number Theory")

@register_task("math500_prealgebra")
class MATH500PrealgebraTask(MATH500BaseTask):
    preprocessing=lambda dataset: dataset.filter(lambda x: x["subject"] == "Prealgebra")

@register_task("math500_precalculus")
class MATH500PrecalculusTask(MATH500BaseTask):
    preprocessing=lambda dataset: dataset.filter(lambda x: x["subject"] == "Precalculus")