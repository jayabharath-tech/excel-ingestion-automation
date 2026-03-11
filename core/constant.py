"""Module to maintain constants"""
from enum import Enum

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

SA_PARTICLES = ["van", "der", "den", "de", "du", "le"]

# Column Names
# TODO: Extract as constants