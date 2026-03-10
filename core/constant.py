"""Module to maintain constants"""
from enum import Enum

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


# Column Names
# TODO: Extract as constants