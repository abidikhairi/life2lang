from enum import Enum


class Task(Enum):
    FAMILY_GENERATION = "family_generation"
    FAMILY_CLASSIFICATION = "family_classification"
    BIOPROCESS_PREDICTION = "bioprocess_prediction"
    LOCALIZATION_PREDICTION = "localization_prediction"
    FUNCTION_PREDICTION = "function_prediction"


TASK_PREFIXES = {
    Task.FAMILY_GENERATION: "design protein sequence for family:",
    Task.FAMILY_CLASSIFICATION: "predict protein family from sequence:",
    Task.BIOPROCESS_PREDICTION: "predict biological processes from protein sequence:",
    Task.LOCALIZATION_PREDICTION: "predict protein localization sites:",
    Task.FUNCTION_PREDICTION: "predict molecular functions from protein sequence:",
}
