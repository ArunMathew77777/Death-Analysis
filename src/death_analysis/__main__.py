"""for main"""
from .eda import EdaAnalysis
from .inference import InferenceAnalysis

"""
Here we are calling different functions through main so that it could work with -m command. here we are calling the both things eda and inf.
"""


def main():
    print("hello professor Molin")
    eda()
    inf()


"""Here we are defining the eda part """


def eda():
    print("EDA start---------------------------------------")

    EdaAnalysis.trend_all_cause()
    EdaAnalysis.temp_trend()
    EdaAnalysis.lead_cause()
    EdaAnalysis.external_factors()
    EdaAnalysis.natural_by_month()
    EdaAnalysis.monthly_death_disease()
    EdaAnalysis.correlation_matrix()
    EdaAnalysis.all_cause_mortality()
    EdaAnalysis.monthly_all_cause_mortality()
    EdaAnalysis.monthly_heart_cause_morality()

    print("EDA ends----------------------------------------")


""" Here we are calling the inference file"""


def inf():
    print("inf start---------------------------------------")
    InferenceAnalysis.research_question1()
    InferenceAnalysis.research_question2()
    InferenceAnalysis.prediction_model()
    InferenceAnalysis.model_prediction_graph()
    InferenceAnalysis.rq1_model()
    InferenceAnalysis.rq2_model()
    InferenceAnalysis.dataModeling()
    InferenceAnalysis.modelQuestion()
    InferenceAnalysis.performClusteringVisualization()
    print("inf ends----------------------------------------")


main()
