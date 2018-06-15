IF [%1] == [] GOTO noimage
python local.py --shape-predictor shape_predictor_68_face_landmarks.dat --image %1
GOTO end
:noimage
python local.py --shape-predictor shape_predictor_68_face_landmarks.dat
:end