# File: modules/face_analysis.py

from deepface import DeepFace


def analyze_gender(face_crop, enforce_detection=False):
    """
    Analyze the gender of a cropped face image using DeepFace.

    Returns:
        str: Predicted gender ('Man', 'Woman', or 'Unknown')
    """
    try:
        result = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=enforce_detection)
        gender_dict = result[0]['gender']
        return max(gender_dict, key=gender_dict.get)
    except Exception as e:
        print(f"[FaceAnalysis] Gender detection error: {e}")
        return "Unknown"


def analyze_attributes(face_crop, actions=['gender', 'emotion', 'age'], enforce_detection=False):
    """
    Analyze facial attributes using DeepFace.

    Args:
        face_crop: Cropped face image
        actions (list): List of attributes to analyze (gender, emotion, age, race)
        enforce_detection (bool): Whether to raise error if no face is found

    Returns:
        dict: Analysis results or {'error': str}
    """
    try:
        result = DeepFace.analyze(face_crop, actions=actions, enforce_detection=enforce_detection)
        return result[0] if isinstance(result, list) else result
    except Exception as e:
        print(f"[FaceAnalysis] Attribute analysis error: {e}")
        return {'error': str(e)}
