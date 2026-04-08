from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="flask_db"
)
mycursor = mydb.cursor()
last_detected_name = ""
last_detected_status = ""
last_detected_message = ""
last_detected_category = ""
last_detected_ballot_count = 0
last_detected_ballot_labels = []
last_detected_time = 0
today_scans = []


def preprocess_face(face_img):
    resized_face = cv2.resize(face_img, (200, 200))
    return cv2.equalizeHist(resized_face)


def get_ballot_details(voter_category):
    ballot_map = {
        "tetap": {
            "count": 5,
            "labels": [
                "Surat suara presiden",
                "Surat suara DPR",
                "Surat suara DPD",
                "Surat suara DPRD provinsi",
                "Surat suara DPRD kabupaten kota"
            ]
        },
        "antar_provinsi": {
            "count": 1,
            "labels": [
                "Surat suara presiden"
            ]
        },
        "antar_kabkota": {
            "count": 4,
            "labels": [
                "Surat suara presiden",
                "Surat suara DPR",
                "Surat suara DPD",
                "Surat suara DPRD provinsi"
            ]
        }
    }

    return ballot_map.get(voter_category, {"count": 0, "labels": []})


def get_voter_category_label(voter_category):
    category_labels = {
        "tetap": "Pemilih Tetap",
        "antar_provinsi": "Pemilih Pindahan Antar Provinsi",
        "antar_kabkota": "Pemilih Pindahan Antar Kabupaten Kota"
    }

    return category_labels.get(voter_category, voter_category)


def is_already_scanned(person_id):
    return any(scan[1] == person_id for scan in today_scans)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "C:/laragon/www/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
    dataset_dir = os.path.join(app.root_path, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.2, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5

        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)

    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]

    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0

    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.cvtColor(face_cropped(img), cv2.COLOR_BGR2GRAY)
            face = preprocess_face(face)

            file_name = nbr + "." + str(img_id) + ".jpg"
            file_name_path = os.path.join(dataset_dir, file_name)
            db_file_path = "dataset/" + file_name
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            mycursor.execute(
                """INSERT INTO `img_dataset` (`img_id`, `img_person`, `img_path`) VALUES (%s, %s, %s)""",
                (img_id, nbr, db_file_path)
            )
            mydb.commit()

            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break

    cap.release()
    cv2.destroyAllWindows()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "C:/laragon/www/FlaskOpencv_FaceRecognition/dataset"

    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        imageNp = preprocess_face(imageNp)
        id = int(os.path.split(image)[1].split(".")[0])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect('../')


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        global last_detected_name, last_detected_status, last_detected_message
        global last_detected_category, last_detected_ballot_count
        global last_detected_ballot_labels, last_detected_time, today_scans
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            face_region = gray_image[y:y + h, x:x + w]
            face_region = preprocess_face(face_region)
            id, pred = clf.predict(face_region)
            confidence = int(100 * (1 - pred / 300))

            mycursor.execute("select b.prs_name, b.prs_skill "
                             "  from img_dataset a "
                              "  left join prs_mstr b on a.img_person = b.prs_nbr "
                             " where a.img_person = " + str(id) +
                             " limit 1")
            voter_data = mycursor.fetchone()

            if not voter_data:
                continue

            s = voter_data[0]
            voter_category = voter_data[1]
            voter_category_label = get_voter_category_label(voter_category)
            ballot_details = get_ballot_details(voter_category)

            if confidence > 70:
                cv2.putText(img, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                current_time = time.time()
                if s != last_detected_name or current_time - last_detected_time > 5:
                    last_detected_name = s
                    last_detected_status = "known"
                    last_detected_message = s + " terverifikasi"
                    last_detected_category = voter_category_label
                    last_detected_ballot_count = ballot_details["count"]
                    last_detected_ballot_labels = ballot_details["labels"]
                    last_detected_time = current_time
                    if not is_already_scanned(id):
                        today_scans.insert(0, [
                            len(today_scans) + 1,
                            id,
                            s,
                            "Terverifikasi",
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ])
                        today_scans = today_scans[:50]
            else:
                cv2.putText(img, "Tidak Terdaftar", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                current_time = time.time()
                if last_detected_status != "unknown" or current_time - last_detected_time > 5:
                    last_detected_name = "Tidak Dikenal"
                    last_detected_status = "unknown"
                    last_detected_message = "Tidak Terdaftar Pemilih"
                    last_detected_category = ""
                    last_detected_ballot_count = 0
                    last_detected_ballot_labels = []
                    last_detected_time = current_time

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.2, 8, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(
        "C:/laragon/www/FlaskOpencv_FaceRecognition/resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    wCam, hCam = 500, 400

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

@app.route('/')
def home():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()

    return render_template('index.html', data=data)


@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 1) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))

    return render_template('addprsn.html', newnbr=int(nbr))


@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')

    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()

    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))


@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('gendataset.html', prs=prs)


@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/last_detected')
def last_detected():
    return jsonify({
        'name': last_detected_name,
        'status': last_detected_status,
        'message': last_detected_message,
        'category': last_detected_category,
        'ballot_count': last_detected_ballot_count,
        'ballot_labels': last_detected_ballot_labels,
        'timestamp': last_detected_time
    })


@app.route('/countTodayScan')
def count_today_scan():
    return jsonify({
        'rowcount': len(today_scans)
    })


@app.route('/loadData')
def load_data():
    return jsonify({
        'rowcount': len(today_scans),
        'data': today_scans
    })


@app.route('/fr_page')
def fr_page():
    return render_template('fr_page.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
