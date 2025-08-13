from deepface import DeepFace
import cv2

def analyze_gender(face_crop, enforce_detection=False):
    """Analyze the gender of a cropped face image using DeepFace."""
    try:
        if face_crop is None:
            return "Unknown"

        # Ensure RGB format
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=enforce_detection)

        # if isinstance(result, list):
        #     result = result[0]
     
        if isinstance(result, list):
            gender_dict = result[0]['gender']
        else:
            gender_dict = result['gender']


        gender_dict = result.get('gender', {})
        return result.get('dominant_gender', max(gender_dict, key=gender_dict.get)) if gender_dict else "Unknown"

    except Exception as e:
        print(f"[FaceAnalysis] Gender detection error: {e}")
        return "Unknown"


def analyze_attributes(face_crop, actions=['gender', 'emotion', 'age'], enforce_detection=False):
    """Analyze facial attributes using DeepFace."""
    try:
        if face_crop is None:
            return {'error': 'No face crop provided'}

        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(face_crop, actions=actions, enforce_detection=enforce_detection)

        if isinstance(result, list):
            result = result[0]

        return result

    except Exception as e:
        print(f"[FaceAnalysis] Attribute analysis error: {e}")
        return {'error': str(e)}
