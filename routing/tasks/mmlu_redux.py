from yeval.task import register_task, YevalTask
from yeval.log.usage import log_token_usage

def input_fn(x):
    letter_choice = ["A", "B", "C", "D"]
    text_choice = x["choices"]
    choice_list = [f"{a}. {b}" for a,b in list(zip(letter_choice, text_choice))]
    question = x['question']
    choice_string = '\n'.join(choice_list)
    return f"Answer with either A, B, C, or D.\nQuestion:\n{question}\n{choice_string}\nAnswer:"

def output_fn(x):
    letter_choice = ["A", "B", "C", "D"]
    text_choice = x["choices"]
    label = x['answer']
    text = text_choice[label]
    letter = letter_choice[label]
    return [letter, text, f"{letter}. {text}"]

def eval_fn(prediction, ground_truth):
    score = 0
    try:
        letter, text, full_span = ground_truth
        if full_span == ground_truth:
            return 1
        prediction = prediction.split(".")[0]
        if prediction in ["A", "B", "C", "D"]:
            if prediction == letter:
                score = 1
        elif prediction == text:
            score = 1
    except Exception as e:
        pass
    return score

class MMLURedux(YevalTask):
    data_path="edinburgh-dawg/mmlu-redux-2.0"
    input_text=input_fn
    output_text=output_fn
    test_split="test"
    evaluation={"accuracy": eval_fn}

@register_task("train:mmlu_redux_abstract_algebra")
class MMLUReduxAbstractAlgebra(MMLURedux):
    data_name = "abstract_algebra"

@register_task("train:mmlu_redux_anatomy")
class MMLUReduxAnatomy(MMLURedux):
    data_name = "anatomy"

@register_task("train:mmlu_redux_astronomy")
class MMLUReduxAstronomy(MMLURedux):
    data_name = "astronomy"

@register_task("train:mmlu_redux_business_ethics")
class MMLUReduxBusinessEthics(MMLURedux):
    data_name = "business_ethics"

@register_task("train:mmlu_redux_clinical_knowledge")
class MMLUReduxClinicalKnowledge(MMLURedux):
    data_name = "clinical_knowledge"

@register_task("train:mmlu_redux_college_biology")
class MMLUReduxCollegeBiology(MMLURedux):
    data_name = "college_biology"

@register_task("train:mmlu_redux_college_chemistry")
class MMLUReduxCollegeChemistry(MMLURedux):
    data_name = "college_chemistry"

@register_task("train:mmlu_redux_college_computer_science")
class MMLUReduxCollegeComputerScience(MMLURedux):
    data_name = "college_computer_science"

@register_task("train:mmlu_redux_college_mathematics")
class MMLUReduxCollegeMathematics(MMLURedux):
    data_name = "college_mathematics"

@register_task("train:mmlu_redux_college_medicine")
class MMLUReduxCollegeMedicine(MMLURedux):
    data_name = "college_medicine"

@register_task("train:mmlu_redux_college_physics")
class MMLUReduxCollegePhysics(MMLURedux):
    data_name = "college_physics"

@register_task("train:mmlu_redux_computer_security")
class MMLUReduxComputerSecurity(MMLURedux):
    data_name = "computer_security"

@register_task("train:mmlu_redux_conceptual_physics")
class MMLUReduxConceptualPhysics(MMLURedux):
    data_name = "conceptual_physics"

@register_task("train:mmlu_redux_econometrics")
class MMLUReduxEconometrics(MMLURedux):
    data_name = "econometrics"

@register_task("train:mmlu_redux_electrical_engineering")
class MMLUReduxElectricalEngineering(MMLURedux):
    data_name = "electrical_engineering"

@register_task("train:mmlu_redux_elementary_mathematics")
class MMLUReduxElementaryMathematics(MMLURedux):
    data_name = "elementary_mathematics"

@register_task("train:mmlu_redux_formal_logic")
class MMLUReduxFormalLogic(MMLURedux):
    data_name = "formal_logic"

@register_task("train:mmlu_redux_global_facts")
class MMLUReduxGlobalFacts(MMLURedux):
    data_name = "global_facts"

@register_task("train:mmlu_redux_high_school_biology")
class MMLUReduxHighSchoolBiology(MMLURedux):
    data_name = "high_school_biology"

@register_task("train:mmlu_redux_high_school_chemistry")
class MMLUReduxHighSchoolChemistry(MMLURedux):
    data_name = "high_school_chemistry"

@register_task("train:mmlu_redux_high_school_computer_science")
class MMLUReduxHighSchoolComputerScience(MMLURedux):
    data_name = "high_school_computer_science"

@register_task("train:mmlu_redux_high_school_european_history")
class MMLUReduxHighSchoolEuropeanHistory(MMLURedux):
    data_name = "high_school_european_history"

@register_task("train:mmlu_redux_high_school_geography")
class MMLUReduxHighSchoolGeography(MMLURedux):
    data_name = "high_school_geography"

@register_task("train:mmlu_redux_high_school_government_and_politics")
class MMLUReduxHighSchoolGovernmentAndPolitics(MMLURedux):
    data_name = "high_school_government_and_politics"

@register_task("train:mmlu_redux_high_school_macroeconomics")
class MMLUReduxHighSchoolMacroeconomics(MMLURedux):
    data_name = "high_school_macroeconomics"

@register_task("train:mmlu_redux_high_school_mathematics")
class MMLUReduxHighSchoolMathematics(MMLURedux):
    data_name = "high_school_mathematics"

@register_task("train:mmlu_redux_high_school_microeconomics")
class MMLUReduxHighSchoolMicroeconomics(MMLURedux):
    data_name = "high_school_microeconomics"

@register_task("train:mmlu_redux_high_school_physics")
class MMLUReduxHighSchoolPhysics(MMLURedux):
    data_name = "high_school_physics"

@register_task("train:mmlu_redux_high_school_psychology")
class MMLUReduxHighSchoolPsychology(MMLURedux):
    data_name = "high_school_psychology"

@register_task("train:mmlu_redux_high_school_statistics")
class MMLUReduxHighSchoolStatistics(MMLURedux):
    data_name = "high_school_statistics"

@register_task("train:mmlu_redux_high_school_us_history")
class MMLUReduxHighSchoolUSHistory(MMLURedux):
    data_name = "high_school_us_history"

@register_task("train:mmlu_redux_high_school_world_history")
class MMLUReduxHighSchoolWorldHistory(MMLURedux):
    data_name = "high_school_world_history"

@register_task("train:mmlu_redux_human_aging")
class MMLUReduxHumanAging(MMLURedux):
    data_name = "human_aging"

@register_task("train:mmlu_redux_human_sexuality")
class MMLUReduxHumanSexuality(MMLURedux):
    data_name = "human_sexuality"

@register_task("train:mmlu_redux_international_law")
class MMLUReduxInternationalLaw(MMLURedux):
    data_name = "international_law"

@register_task("train:mmlu_redux_jurisprudence")
class MMLUReduxJurisprudence(MMLURedux):
    data_name = "jurisprudence"

@register_task("train:mmlu_redux_logical_fallacies")
class MMLUReduxLogicalFallacies(MMLURedux):
    data_name = "logical_fallacies"

@register_task("train:mmlu_redux_machine_learning")
class MMLUReduxMachineLearning(MMLURedux):
    data_name = "machine_learning"

@register_task("train:mmlu_redux_management")
class MMLUReduxManagement(MMLURedux):
    data_name = "management"

@register_task("train:mmlu_redux_marketing")
class MMLUReduxMarketing(MMLURedux):
    data_name = "marketing"

@register_task("train:mmlu_redux_medical_genetics")
class MMLUReduxMedicalGenetics(MMLURedux):
    data_name = "medical_genetics"

@register_task("train:mmlu_redux_miscellaneous")
class MMLUReduxMiscellaneous(MMLURedux):
    data_name = "miscellaneous"

@register_task("train:mmlu_redux_moral_disputes")
class MMLUReduxMoralDisputes(MMLURedux):
    data_name = "moral_disputes"

@register_task("train:mmlu_redux_moral_scenarios")
class MMLUReduxMoralScenarios(MMLURedux):
    data_name = "moral_scenarios"

@register_task("train:mmlu_redux_nutrition")
class MMLUReduxNutrition(MMLURedux):
    data_name = "nutrition"

@register_task("train:mmlu_redux_philosophy")
class MMLUReduxPhilosophy(MMLURedux):
    data_name = "philosophy"

@register_task("train:mmlu_redux_prehistory")
class MMLUReduxPrehistory(MMLURedux):
    data_name = "prehistory"

@register_task("train:mmlu_redux_professional_accounting")
class MMLUReduxProfessionalAccounting(MMLURedux):
    data_name = "professional_accounting"

@register_task("train:mmlu_redux_professional_law")
class MMLUReduxProfessionalLaw(MMLURedux):
    data_name = "professional_law"

@register_task("train:mmlu_redux_professional_medicine")
class MMLUReduxProfessionalMedicine(MMLURedux):
    data_name = "professional_medicine"

@register_task("train:mmlu_redux_professional_psychology")
class MMLUReduxProfessionalPsychology(MMLURedux):
    data_name = "professional_psychology"

@register_task("train:mmlu_redux_public_relations")
class MMLUReduxPublicRelations(MMLURedux):
    data_name = "public_relations"

@register_task("train:mmlu_redux_security_studies")
class MMLUReduxSecurityStudies(MMLURedux):
    data_name = "security_studies"

@register_task("train:mmlu_redux_sociology")
class MMLUReduxSociology(MMLURedux):
    data_name = "sociology"

@register_task("train:mmlu_redux_us_foreign_policy")
class MMLUReduxUSForeignPolicy(MMLURedux):
    data_name = "us_foreign_policy"

@register_task("train:mmlu_redux_virology")
class MMLUReduxVirology(MMLURedux):
    data_name = "virology"

@register_task("train:mmlu_redux_world_religions")
class MMLUReduxWorldReligions(MMLURedux):
    data_name = "world_religions"




if __name__ == "__main__":
    pass
