from enum import Enum


class AnnotationSelection(Enum):
    """
    NOTE: Add color if that becomes a thing
    """
    SHAPES = 0
    CHARACTERS = 1
    SHAPES_AND_CHARACTERS = 2
    TARGETS_ONLY = 3


def shapes_filter(class_name: str) -> str | None:
    """
    Filters out 'background' and single-character names (which are letter labels)
    """
    if class_name.lower() == 'background' or len(class_name) <= 1:
        return None

    return class_name


def characters_filter(class_name: str) -> str | None:
    """
    Filters out 'background' and multi-character names (which are shape labels)
    """
    if class_name.lower() == 'background' or len(class_name) >= 1:
        return None

    return class_name


def shapes_and_characters_filter(class_name: str) -> str | None:
    """
    Filters out 'background'
    """
    if class_name.lower() == 'background':
        return None

    return class_name


def targets_name_transform(class_name: str) -> str | None:
    """
    Converts all shape names to 'target' while filtering out 'background' and character labels.
    """
    if shapes_filter(class_name):
        return 'target'

    return None